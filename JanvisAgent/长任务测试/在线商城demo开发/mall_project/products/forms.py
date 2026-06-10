# encoding: utf-8
# @Time    : 2026/06/09 16:00

from django import forms
from .models import Product


class ProductForm(forms.ModelForm):
    """商品创建/编辑表单"""

    class Meta:
        model = Product
        fields = ("name", "description", "price", "stock", "is_active")
        labels = {
            "name": "商品名称",
            "description": "商品描述",
            "price": "价格",
            "stock": "库存",
            "is_active": "是否上架",
        }
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
            "price": forms.NumberInput(attrs={"min": 0, "step": "0.01"}),
            "stock": forms.NumberInput(attrs={"min": 0}),
        }
