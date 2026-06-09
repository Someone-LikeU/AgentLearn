# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.test import TestCase, Client
from django.urls import reverse
from decimal import Decimal
from accounts.models import User, Wallet
from products.models import Product
from cart.models import CartItem
from .models import Order, OrderItem


class OrderTest(TestCase):
    """订单测试"""

    def setUp(self):
        self.client = Client()
        self.customer = User.objects.create_user(
            username="customer",
            password="testpass123",
            user_type=User.UserType.CUSTOMER,
        )
        self.customer_wallet = Wallet.objects.create(
            user=self.customer,
            balance=Decimal("5000.00"),
        )
        self.seller = User.objects.create_user(
            username="seller",
            password="testpass123",
            user_type=User.UserType.SELLER,
        )
        self.seller_wallet = Wallet.objects.create(
            user=self.seller,
            balance=Decimal("0.00"),
        )
        self.product = Product.objects.create(
            name="测试商品",
            description="测试描述",
            price=Decimal("100.00"),
            stock=10,
            seller=self.seller,
            is_active=True,
        )

    def test_buy_now(self):
        """测试立即购买"""
        self.client.login(username="customer", password="testpass123")
        response = self.client.post(
            reverse("orders:buy_now", kwargs={"product_id": self.product.pk}),
            {"quantity": 2},
        )
        self.assertEqual(response.status_code, 302)

        # 检查订单创建
        self.assertEqual(Order.objects.count(), 1)
        order = Order.objects.first()
        self.assertEqual(order.total_amount, Decimal("200.00"))

        # 检查客户余额扣减
        self.customer_wallet.refresh_from_db()
        self.assertEqual(self.customer_wallet.balance, Decimal("4800.00"))

        # 检查商家余额增加（扣除3%手续费）
        self.seller_wallet.refresh_from_db()
        expected_income = Decimal("200.00") * Decimal("0.97")
        self.assertEqual(self.seller_wallet.balance, expected_income)

        # 检查库存减少
        self.product.refresh_from_db()
        self.assertEqual(self.product.stock, 8)

    def test_buy_now_insufficient_balance(self):
        """测试余额不足购买"""
        self.customer_wallet.balance = Decimal("50.00")
        self.customer_wallet.save()
        self.client.login(username="customer", password="testpass123")
        response = self.client.post(
            reverse("orders:buy_now", kwargs={"product_id": self.product.pk}),
            {"quantity": 2},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Order.objects.count(), 0)

    def test_checkout_cart(self):
        """测试购物车结算"""
        self.client.login(username="customer", password="testpass123")
        CartItem.objects.create(
            user=self.customer,
            product=self.product,
            quantity=3,
        )
        response = self.client.post(reverse("orders:checkout"))
        self.assertEqual(response.status_code, 302)

        # 检查订单创建
        self.assertEqual(Order.objects.count(), 1)
        order = Order.objects.first()
        self.assertEqual(order.total_amount, Decimal("300.00"))

        # 检查购物车清空
        self.assertEqual(CartItem.objects.count(), 0)

        # 检查余额
        self.customer_wallet.refresh_from_db()
        self.assertEqual(self.customer_wallet.balance, Decimal("4700.00"))

    def test_fee_calculation(self):
        """测试手续费计算"""
        self.client.login(username="customer", password="testpass123")
        self.client.post(
            reverse("orders:buy_now", kwargs={"product_id": self.product.pk}),
            {"quantity": 1},
        )
        self.seller_wallet.refresh_from_db()
        # 100 * 0.97 = 97
        self.assertEqual(self.seller_wallet.balance, Decimal("97.00"))
