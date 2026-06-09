# encoding: utf-8
# @Time    : 2026/06/09 11:46

import random
from decimal import Decimal
from django.core.management.base import BaseCommand
from accounts.models import User, Wallet
from products.models import Product


class Command(BaseCommand):
    help = "初始化测试数据：管理员、3个商家、2个客户、随机商品"

    def handle(self, *args, **options):
        self.stdout.write("开始初始化数据...")

        # 创建管理员账号
        admin_user, created = User.objects.get_or_create(
            username="admin",
            defaults={
                "email": "admin@example.com",
                "user_type": User.UserType.CUSTOMER,
                "is_superuser": True,
                "is_staff": True,
            },
        )
        admin_user.set_password("admin123")
        admin_user.is_superuser = True
        admin_user.is_staff = True
        admin_user.save()
        if created:
            self.stdout.write("  创建管理员: admin (密码: admin123)")
        else:
            self.stdout.write("  管理员已存在，已重置密码为: admin123")

        # 创建3个商家
        sellers_data = [
            {"username": "seller1", "email": "seller1@example.com", "password": "test123456"},
            {"username": "seller2", "email": "seller2@example.com", "password": "test123456"},
            {"username": "seller3", "email": "seller3@example.com", "password": "test123456"},
        ]

        sellers = []
        for data in sellers_data:
            user, created = User.objects.get_or_create(
                username=data["username"],
                defaults={
                    "email": data["email"],
                    "user_type": User.UserType.SELLER,
                },
            )
            if created:
                user.set_password(data["password"])
                user.save()
                self.stdout.write(f"  创建商家: {data['username']}")
            else:
                self.stdout.write(f"  商家已存在: {data['username']}")
            sellers.append(user)

        # 创建2个客户
        customers_data = [
            {"username": "customer1", "email": "customer1@example.com", "password": "test123456"},
            {"username": "customer2", "email": "customer2@example.com", "password": "test123456"},
        ]

        for data in customers_data:
            user, created = User.objects.get_or_create(
                username=data["username"],
                defaults={
                    "email": data["email"],
                    "user_type": User.UserType.CUSTOMER,
                },
            )
            if created:
                user.set_password(data["password"])
                user.save()
                # 创建钱包，赠送5000软妹币
                Wallet.objects.create(user=user, balance=Decimal("5000.00"))
                self.stdout.write(f"  创建客户: {data['username']} (赠送5000软妹币)")
            else:
                self.stdout.write(f"  客户已存在: {data['username']}")

        # 商品数据
        products_data = [
            # seller1 的商品
            {"name": "iPhone 15 Pro", "description": "苹果最新旗舰手机，A17 Pro芯片", "price": 8999, "stock": 50, "seller": sellers[0]},
            {"name": "MacBook Air M3", "description": "轻薄笔记本，M3芯片，超长续航", "price": 9499, "stock": 30, "seller": sellers[0]},
            {"name": "AirPods Pro 2", "description": "主动降噪无线耳机", "price": 1899, "stock": 100, "seller": sellers[0]},
            # seller2 的商品
            {"name": "小米14 Ultra", "description": "徕卡影像旗舰，骁龙8 Gen3", "price": 6499, "stock": 40, "seller": sellers[1]},
            {"name": "华为MatePad Pro", "description": "专业平板电脑，鸿蒙系统", "price": 4999, "stock": 25, "seller": sellers[1]},
            {"name": "索尼WH-1000XM5", "description": "顶级降噪头戴式耳机", "price": 2699, "stock": 60, "seller": sellers[1]},
            {"name": "任天堂Switch OLED", "description": "便携式游戏机，OLED屏幕", "price": 2599, "stock": 80, "seller": sellers[1]},
            # seller3 的商品
            {"name": "戴森V15吸尘器", "description": "无线吸尘器，强劲吸力", "price": 5490, "stock": 20, "seller": sellers[2]},
            {"name": "Kindle Paperwhite", "description": "电子书阅读器，护眼屏幕", "price": 1068, "stock": 150, "seller": sellers[2]},
        ]

        for data in products_data:
            product, created = Product.objects.get_or_create(
                name=data["name"],
                defaults={
                    "description": data["description"],
                    "price": Decimal(str(data["price"])),
                    "stock": data["stock"],
                    "seller": data["seller"],
                    "is_active": True,
                },
            )
            if created:
                self.stdout.write(f"  创建商品: {data['name']} (商家: {data['seller'].username})")
            else:
                self.stdout.write(f"  商品已存在: {data['name']}")

        self.stdout.write(self.style.SUCCESS("数据初始化完成！"))
