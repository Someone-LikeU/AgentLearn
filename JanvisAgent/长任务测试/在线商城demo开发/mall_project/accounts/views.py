# encoding: utf-8
# @Time    : 2026/06/09 11:46

import time
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db import models
from decimal import Decimal
from .forms import UserRegisterForm, UserLoginForm
from .models import User, Wallet
from orders.models import OrderItem

# 登录失败记录：{username: {"count": int, "locked_until": float}}
_login_failures = {}

MAX_LOGIN_ATTEMPTS = 5
LOCK_DURATION = 30 * 60  # 30分钟


def _is_locked(username):
    """检查用户是否被锁定"""
    record = _login_failures.get(username)
    if record and record.get("locked_until"):
        if time.time() < record["locked_until"]:
            remaining = int(record["locked_until"] - time.time())
            return True, remaining
        else:
            # 锁定已过期，清除记录
            _login_failures.pop(username, None)
    return False, 0


def _record_failure(username):
    """记录一次登录失败"""
    if username not in _login_failures:
        _login_failures[username] = {"count": 0, "locked_until": 0}
    _login_failures[username]["count"] += 1
    if _login_failures[username]["count"] >= MAX_LOGIN_ATTEMPTS:
        _login_failures[username]["locked_until"] = time.time() + LOCK_DURATION


def _clear_failures(username):
    """登录成功后清除失败记录"""
    _login_failures.pop(username, None)


def register_view(request):
    """用户注册视图"""
    if request.user.is_authenticated:
        return redirect("products:product_list")

    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # 如果是客户，创建钱包并赠送5000软妹币
            if user.is_customer():
                Wallet.objects.create(
                    user=user,
                    balance=Decimal("5000.00"),
                )
            messages.success(request, "注册成功！")
            login(request, user)
            return redirect("products:product_list")
    else:
        form = UserRegisterForm()

    return render(request, "accounts/register.html", {"form": form})


def login_view(request):
    """用户登录视图"""
    if request.user.is_authenticated:
        return redirect("products:product_list")

    login_error = None
    remaining_attempts = None

    if request.method == "POST":
        form = UserLoginForm(request, data=request.POST)
        username = request.POST.get("username", "")

        # 检查是否被锁定
        locked, remaining = _is_locked(username)
        if locked:
            login_error = f"账户已锁定，请在 {remaining // 60} 分 {remaining % 60} 秒后重试"
            remaining_attempts = 0
        elif form.is_valid():
            user = form.get_user()
            _clear_failures(username)
            login(request, user)
            messages.success(request, "登录成功！")
            if user.is_superuser:
                return redirect("accounts:admin_dashboard")
            return redirect("products:product_list")
        else:
            _record_failure(username)
            record = _login_failures.get(username, {})
            attempts = record.get("count", 0)

            # 再次检查是否刚被锁定
            locked2, remaining2 = _is_locked(username)
            if locked2:
                login_error = f"账户已锁定，请在 {remaining2 // 60} 分 {remaining2 % 60} 秒后重试"
                remaining_attempts = 0
            else:
                login_error = "用户名或密码错误"
                remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts
    else:
        form = UserLoginForm()

    return render(request, "accounts/login.html", {
        "form": form,
        "login_error": login_error,
        "remaining_attempts": remaining_attempts,
    })


def logout_view(request):
    """用户登出视图"""
    logout(request)
    messages.success(request, "已退出登录")
    return redirect("accounts:login")


@login_required
def profile_view(request):
    """用户个人信息视图"""
    wallet = None
    if request.user.is_customer():
        wallet = getattr(request.user, "wallet", None)

    # 处理充值请求
    if request.method == "POST" and request.user.is_customer():
        amount = request.POST.get("amount")
        if amount:
            try:
                amount_decimal = Decimal(amount)
                if amount_decimal > 0:
                    if not wallet:
                        wallet = Wallet.objects.create(user=request.user, balance=Decimal("0.00"))
                    wallet.add(amount_decimal)
                    messages.success(request, f"充值成功！已充值 {amount_decimal} 软妹币")
                    return redirect("accounts:profile")
                else:
                    messages.error(request, "充值金额必须大于0")
            except Exception:
                messages.error(request, "充值金额格式错误")
        else:
            messages.error(request, "请输入充值金额")

    return render(request, "accounts/profile.html", {"wallet": wallet})


@login_required
def admin_dashboard(request):
    """管理员界面"""
    if not request.user.is_superuser:
        messages.error(request, "无权访问")
        return redirect("products:product_list")

    sellers = User.objects.filter(user_type=User.UserType.SELLER).prefetch_related("products")
    customers = User.objects.filter(user_type=User.UserType.CUSTOMER, is_superuser=False).select_related("wallet")

    # 为每个商家计算商品销量和手续费
    seller_data = []
    for seller in sellers:
        products = seller.products.all()
        product_list = []
        total_fee = Decimal("0.00")
        for product in products:
            sold_quantity = OrderItem.objects.filter(product=product).aggregate(
                total=models.Sum("quantity")
            )["total"] or 0
            fee = product.price * sold_quantity * Decimal("0.03")
            total_fee += fee
            product_list.append({
                "name": product.name,
                "price": product.price,
                "sold_quantity": sold_quantity,
                "fee": fee,
            })
        seller_data.append({
            "username": seller.username,
            "products": product_list,
            "total_fee": total_fee,
        })

    # 客户余额用***表示
    customer_data = []
    for customer in customers:
        wallet = getattr(customer, "wallet", None)
        customer_data.append({
            "username": customer.username,
            "balance": "***",
        })

    return render(request, "accounts/admin_dashboard.html", {
        "seller_data": seller_data,
        "customer_data": customer_data,
    })
