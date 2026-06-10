# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_POST
from .models import CartItem
from products.models import Product


@login_required
def cart_view(request):
    """购物车视图"""
    if not request.user.is_customer():
        messages.error(request, "只有客户可以访问购物车")
        return redirect("products:product_list")

    cart_items = CartItem.objects.filter(user=request.user).select_related("product")
    total = sum(item.get_total_price() for item in cart_items)

    return render(request, "cart/cart.html", {
        "cart_items": cart_items,
        "total": total,
    })


@login_required
@require_POST
def add_to_cart(request, product_id):
    """添加商品到购物车"""
    if not request.user.is_customer():
        messages.error(request, "只有客户可以添加商品到购物车")
        return redirect("products:product_list")

    product = get_object_or_404(Product, pk=product_id, is_active=True)
    quantity = int(request.POST.get("quantity", 1))

    if quantity > product.stock:
        messages.error(request, "库存不足")
        return redirect("products:product_detail", pk=product_id)

    cart_item, created = CartItem.objects.get_or_create(
        user=request.user,
        product=product,
        defaults={"quantity": quantity},
    )

    if not created:
        new_quantity = cart_item.quantity + quantity
        if new_quantity > product.stock:
            messages.error(request, "库存不足")
            return redirect("products:product_detail", pk=product_id)
        cart_item.quantity = new_quantity
        cart_item.save()

    messages.success(request, "商品已添加到购物车")
    return redirect("cart:cart")


@login_required
@require_POST
def update_cart_item(request, item_id):
    """更新购物车商品数量"""
    cart_item = get_object_or_404(CartItem, pk=item_id, user=request.user)
    quantity = int(request.POST.get("quantity", 1))

    if quantity <= 0:
        cart_item.delete()
        messages.success(request, "商品已从购物车移除")
    elif quantity > cart_item.product.stock:
        messages.error(request, "库存不足")
    else:
        cart_item.quantity = quantity
        cart_item.save()
        messages.success(request, "数量已更新")

    return redirect("cart:cart")


@login_required
@require_POST
def remove_from_cart(request, item_id):
    """从购物车移除商品"""
    cart_item = get_object_or_404(CartItem, pk=item_id, user=request.user)
    cart_item.delete()
    messages.success(request, "商品已从购物车移除")
    return redirect("cart:cart")
