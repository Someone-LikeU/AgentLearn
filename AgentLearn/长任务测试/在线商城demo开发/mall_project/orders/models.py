# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.db import models
from django.conf import settings
from decimal import Decimal


class Order(models.Model):
    """订单模型"""

    class Status(models.TextChoices):
        PENDING = "pending", "待支付"
        COMPLETED = "completed", "已完成"
        CANCELLED = "cancelled", "已取消"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="orders",
        verbose_name="用户",
    )
    total_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        verbose_name="总金额",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.COMPLETED,
        verbose_name="订单状态",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "订单"
        verbose_name_plural = "订单"
        ordering = ["-created_at"]

    def __str__(self):
        return f"订单 #{self.pk} - {self.user.username}"


class OrderItem(models.Model):
    """订单项"""

    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name="items",
        verbose_name="订单",
    )
    product = models.ForeignKey(
        "products.Product",
        on_delete=models.SET_NULL,
        null=True,
        verbose_name="商品",
    )
    quantity = models.PositiveIntegerField(verbose_name="数量")
    price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        verbose_name="单价",
    )

    class Meta:
        verbose_name = "订单项"
        verbose_name_plural = "订单项"

    def __str__(self):
        product_name = self.product.name if self.product else "已删除商品"
        return f"{product_name} x{self.quantity}"

    def get_subtotal(self):
        return self.price * self.quantity
