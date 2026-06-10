# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.contrib.auth.models import AbstractUser
from django.db import models
from decimal import Decimal


class User(AbstractUser):
    """扩展Django用户模型，支持商家和客户两种角色"""

    class UserType(models.TextChoices):
        CUSTOMER = "customer", "客户"
        SELLER = "seller", "商家"

    user_type = models.CharField(
        max_length=10,
        choices=UserType.choices,
        default=UserType.CUSTOMER,
        verbose_name="用户类型",
    )

    class Meta:
        verbose_name = "用户"
        verbose_name_plural = "用户"

    def is_seller(self):
        return self.user_type == self.UserType.SELLER

    def is_customer(self):
        return self.user_type == self.UserType.CUSTOMER


class Wallet(models.Model):
    """虚拟货币账户"""

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="wallet",
        verbose_name="用户",
    )
    balance = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal("0.00"),
        verbose_name="余额",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "钱包"
        verbose_name_plural = "钱包"

    def __str__(self):
        return f"{self.user.username} - {self.balance} 软妹币"

    def has_sufficient_balance(self, amount):
        return self.balance >= amount

    def deduct(self, amount):
        """扣除余额"""
        if not self.has_sufficient_balance(amount):
            return False
        self.balance -= amount
        self.save()
        return True

    def add(self, amount):
        """增加余额"""
        self.balance += amount
        self.save()
