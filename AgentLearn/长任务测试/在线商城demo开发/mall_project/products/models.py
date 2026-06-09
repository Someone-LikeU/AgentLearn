# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.db import models
from django.conf import settings


class Product(models.Model):
    """商品模型"""

    name = models.CharField(max_length=200, verbose_name="商品名称")
    description = models.TextField(blank=True, verbose_name="商品描述")
    price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        verbose_name="价格",
    )
    stock = models.PositiveIntegerField(default=0, verbose_name="库存")
    seller = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="products",
        verbose_name="商家",
    )
    is_active = models.BooleanField(default=True, verbose_name="是否上架")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "商品"
        verbose_name_plural = "商品"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def is_in_stock(self):
        return self.stock > 0

    def reduce_stock(self, quantity):
        """减少库存"""
        if self.stock >= quantity:
            self.stock -= quantity
            self.save()
            return True
        return False
