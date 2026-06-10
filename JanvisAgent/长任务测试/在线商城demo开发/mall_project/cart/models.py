# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.db import models
from django.conf import settings
from products.models import Product


class CartItem(models.Model):
    """购物车项"""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="cart_items",
        verbose_name="用户",
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        verbose_name="商品",
    )
    quantity = models.PositiveIntegerField(default=1, verbose_name="数量")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "购物车项"
        verbose_name_plural = "购物车项"
        unique_together = ("user", "product")

    def __str__(self):
        return f"{self.user.username} - {self.product.name} x{self.quantity}"

    def get_total_price(self):
        return self.product.price * self.quantity
