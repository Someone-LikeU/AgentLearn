# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Product
from .forms import ProductForm


def product_list_view(request):
    """商品列表视图"""
    products = Product.objects.filter(is_active=True)
    return render(request, "products/product_list.html", {"products": products})


def product_detail_view(request, pk):
    """商品详情视图"""
    product = get_object_or_404(Product, pk=pk, is_active=True)
    return render(request, "products/product_detail.html", {"product": product})


@login_required
def my_products_view(request):
    """商家查看自己的商品列表"""
    if not request.user.is_seller():
        messages.error(request, "只有商家可以访问此页面")
        return redirect("products:product_list")

    products = Product.objects.filter(seller=request.user)
    form = ProductForm()
    return render(request, "products/my_products.html", {"products": products, "form": form})


@login_required
def create_product(request):
    """商家创建新商品"""
    if not request.user.is_seller():
        messages.error(request, "只有商家可以执行此操作")
        return redirect("products:product_list")

    if request.method == "POST":
        form = ProductForm(request.POST)
        if form.is_valid():
            product = form.save(commit=False)
            product.seller = request.user
            product.save()
            messages.success(request, f"商品 {product.name} 创建成功！")
            return redirect("products:my_products")
    else:
        form = ProductForm()

    return render(request, "products/create_product.html", {"form": form})


@login_required
def toggle_product_status(request, pk):
    """商品上架/下架"""
    if not request.user.is_seller():
        messages.error(request, "只有商家可以执行此操作")
        return redirect("products:product_list")

    product = get_object_or_404(Product, pk=pk, seller=request.user)
    product.is_active = not product.is_active
    product.save()
    status = "上架" if product.is_active else "下架"
    messages.success(request, f"商品已{status}")
    return redirect("my_products")
