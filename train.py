import os
import math
import argparse
from datetime import datetime, timedelta
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
import torch.nn as nn
from kan import KAN  # 导入 KAN 类


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 初始化日志和目录
    total_start_time = time.time()
    log_file = "./training_log.txt"
    os.makedirs("./weights", exist_ok=True)

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(f"=== Training Started at {datetime.now()} ===\n")
            f.write(f"Config: {args}\n\n")

    tb_writer = SummaryWriter()

    # 数据准备
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 修改后的数据预处理（移除了训练集的数据增强）
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(256),          # 仅调整大小
            transforms.CenterCrop(224),      # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = MyDataSet(train_images_path, train_images_label, data_transform["train"])
    val_dataset = MyDataSet(val_images_path, val_images_label, data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        drop_last=True,
        collate_fn=val_dataset.collate_fn
    )

    args.num_classes = 4
    # 模型初始化
    model = create_model(
        num_classes=args.num_classes,
        has_logits=False,
        use_kan=args.use_kan,
        use_biformer=args.use_biformer
    ).to(device)

    # 参数统计
    print("\n===== 参数统计 =====")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M ({trainable_params / total_params:.1%})")

    # 冻结逻辑
    if args.freeze_kan or args.freeze_all:
        for name, param in model.named_parameters():
            if 'kan' in name.lower():
                param.requires_grad_(False)
                print(f"冻结KAN层: {name}")

    if args.freeze_biformer or args.freeze_all:
        for name, param in model.named_parameters():
            if 'biformer' in name.lower() or 'attn.qkv' in name:
                param.requires_grad_(False)
                print(f"冻结BiFormer层: {name}")

    if args.freeze_all:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad_(False)
                print(f"冻结非分类层: {name}")

    # 激活KAN检测代码
    if args.use_kan:
        active_kan_layers = [
            name for name, param in model.named_parameters()
            if 'kan' in name.lower() and param.requires_grad
        ]
        print("✅ 实际可训练的KAN层：" if active_kan_layers else "❌ 所有KAN层已被冻结")
        for name in active_kan_layers:
            print(f"  - {name}")
        kan_params = sum(p.numel() for n, p in model.named_parameters() if 'kan' in n)
        print(f"KAN参数量: {kan_params}/{total_params} ({kan_params / total_params:.1%})")

    # 权重加载
    if args.weights != "":
        weights_dict = torch.load(args.weights, map_location=device)

        # 动态删除冲突键
        del_keys = [k for k in weights_dict.keys()
                    if 'head' in k or 'pre_logits' in k]
        for k in del_keys:
            weights_dict.pop(k, None)

        # 加载权重并初始化新增的KAN参数
        model.load_state_dict(weights_dict, strict=False)

        if args.use_kan:
            for name, param in model.named_parameters():
                if 'kan' in name.lower() and name not in weights_dict:
                    if 'weight' in name or 'coeff' in name:
                        if param.dim() >= 2:
                            nn.init.kaiming_normal_(param, mode='fan_in')
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                    print(f"初始化KAN参数：{name}")

    # 冻结设置
    if args.freeze_layers:
        for name, param in model.named_parameters():
            freeze = (
                    ("head" not in name) and
                    ("pre_logits" not in name) and
                    ("kan" not in name.lower())
            )
            param.requires_grad_(not freeze)

    # 优化器和学习率调度器
    pg_normal = []
    pg_kan = []
    for name, p in model.named_parameters():
        if 'kan' in name.lower() and p.requires_grad:
            pg_kan.append(p)
        elif p.requires_grad:
            pg_normal.append(p)

    optimizer = optim.AdamW([
        {'params': pg_normal, 'lr': args.lr},
        {'params': pg_kan, 'lr': args.kan_lr}
    ], weight_decay=0.05)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # 训练和验证
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            true_num_classes=3
        )
        print(f"[train epoch {epoch}] loss: {train_loss:.4f}, acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            true_num_classes=3
        )
        print(f"[valid epoch {epoch}] loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # 记录日志
        epoch_duration = timedelta(seconds=time.time() - epoch_start_time)
        with open(log_file, "a") as f:
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.4f} | "
                f"Duration: {str(epoch_duration)}\n"
            )

        # TensorBoard记录
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # 保存模型
        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

    # 训练结束
    total_duration = timedelta(seconds=int(time.time() - total_start_time))
    with open(log_file, "a") as f:
        f.write(f"\n=== Training Completed ===\n")
        f.write(f"Total epochs: {args.epochs}\n")
        f.write(f"Total duration: {total_duration}\n")
        f.write(f"Avg time per epoch: {total_duration / args.epochs}\n")

    print(f"\nTraining finished. Total time: {total_duration}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--use-biformer', action='store_true', help='启用BiFormer注意力')
    parser.add_argument('--use-kan', action='store_true', help='启用KAN替代部分MLP')
    parser.add_argument('--freeze-kan', action='store_true', help='冻结KAN参数')
    parser.add_argument('--freeze-biformer', action='store_true', help='冻结所有BiFormer参数')
    parser.add_argument('--freeze-all', action='store_true', help='冻结所有非分类层（包括KAN和BiFormer）')
    parser.add_argument('--freeze-layers', action='store_true', help='是否冻结非分类层')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weights', type=str, default=r'D:\vision_transformer\vit_base_patch16_224_in21k.pth')
    parser.add_argument('--data-path', type=str, default=r"D:\vision_transformer\datasets\train")
    parser.add_argument('--kan-lr', type=float, default=1e-4, help='KAN层学习率')
    parser.add_argument('--verbose', action='store_true', help='显示详细冻结信息')

    opt = parser.parse_args()
    main(opt)


#同时启用 BiFormer 和 KAN
# python train.py --use-biformer --use-kan --num_classes 4 --data-path D:\vision_transformer\datasets\train
#全部冻结
# python train.py --use-kan --use-biformer --freeze-kan --freeze-biformer --num_classes 4 --data-path D:\vision_transformer\datasets\train --batch-size 8 --epochs 50



