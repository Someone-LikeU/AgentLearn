# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db import transaction
from decimal import Decimal
from .models import Order, OrderItem
from cart.models import CartItem
from accounts.models import Wallet
from products.models import Product


FEE_RATE = Decimal("0.03")  # 3%手续费


@login_required
def order_list_view(request):
    """订单列表视图"""
    if request.user.is_seller():
        # 商家查看自己卖出商品的订单
        order_ids = OrderItem.objects.filter(
            product__seller=request.user
        ).values_list("order_id", flat=True).distinct()
        orders = Order.objects.filter(pk__in=order_ids).prefetch_related("items").order_by("-created_at")
    else:
        # 客户查看自己的购买订单
        orders = Order.objects.filter(user=request.user).prefetch_related("items").order_by("-created_at")
    return render(request, "orders/order_list.html", {"orders": orders})


@login_required
def order_detail_view(request, pk):
    """订单详情视图"""
    order = get_object_or_404(Order, pk=pk, user=request.user)
    return render(request, "orders/order_detail.html", {"order": order})


@login_required
def buy_now(request, product_id):
    """立即购买"""
    if not request.user.is_customer():
        messages.error(request, "只有客户可以购买商品")
        return redirect("products:product_list")

    product = get_object_or_404(Product, pk=product_id, is_active=True)
    quantity = int(request.POST.get("quantity", 1))

    if quantity > product.stock:
        messages.error(request, "库存不足")
        return redirect("products:product_detail", pk=product_id)

    total_price = product.price * quantity

    # 检查余额
    wallet = getattr(request.user, "wallet", None)
    if not wallet or not wallet.has_sufficient_balance(total_price):
        messages.error(request, "余额不足，请先充值")
        return redirect("products:product_detail", pk=product_id)

    with transaction.atomic():
        # 扣减客户余额
        wallet.deduct(total_price)

        # 计算商家收入（扣除3%手续费）
        seller_income = total_price * (Decimal("1") - FEE_RATE)

        # 增加商家余额
        seller_wallet, _ = Wallet.objects.get_or_create(
            user=product.seller,
            defaults={"balance": Decimal("0.00")},
        )
        seller_wallet.add(seller_income)

        # 减少库存
        product.reduce_stock(quantity)

        # 创建订单
        order = Order.objects.create(
            user=request.user,
            total_amount=total_price,
            status=Order.Status.COMPLETED,
        )

        # 创建订单项
        OrderItem.objects.create(
            order=order,
            product=product,
            quantity=quantity,
            price=product.price,
        )

    messages.success(request, "购买成功！")
    return redirect("orders:order_detail", pk=order.pk)


@login_required
def checkout_cart(request):
    """购物车结算"""
    if not request.user.is_customer():
        messages.error(request, "只有客户可以结算")
        return redirect("products:product_list")

    cart_items = CartItem.objects.filter(user=request.user).select_related("product")

    if not cart_items.exists():
        messages.error(request, "购物车为空")
        return redirect("cart:cart")

    # 检查库存
    for item in cart_items:
        if item.quantity > item.product.stock:
            messages.error(request, f"商品 {item.product.name} 库存不足")
            return redirect("cart:cart")

    total_price = sum(item.get_total_price() for item in cart_items)

    # 检查余额
    wallet = getattr(request.user, "wallet", None)
    if not wallet or not wallet.has_sufficient_balance(total_price):
        messages.error(request, "余额不足，请先充值")
        return redirect("cart:cart")

    with transaction.atomic():
        # 扣减客户余额
        wallet.deduct(total_price)

        # 创建订单
        order = Order.objects.create(
            user=request.user,
            total_amount=total_price,
            status=Order.Status.COMPLETED,
        )

        # 处理每个购物车项
        for item in cart_items:
            # 计算商家收入
            item_total = item.get_total_price()
            seller_income = item_total * (Decimal("1") - FEE_RATE)

            # 增加商家余额
            seller_wallet, _ = Wallet.objects.get_or_create(
                user=item.product.seller,
                defaults={"balance": Decimal("0.00")},
            )
            seller_wallet.add(seller_income)

            # 减少库存
            item.product.reduce_stock(item.quantity)

            # 创建订单项
            OrderItem.objects.create(
                order=order,
                product=item.product,
                quantity=item.quantity,
                price=item.product.price,
            )

        # 清空购物车
        cart_items.delete()

    messages.success(request, "结算成功！")
    return redirect("orders:order_detail", pk=order.pk)
