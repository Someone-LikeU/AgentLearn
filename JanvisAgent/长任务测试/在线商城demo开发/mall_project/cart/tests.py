# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.test import TestCase, Client
from django.urls import reverse
from decimal import Decimal
from accounts.models import User, Wallet
from products.models import Product
from .models import CartItem


class CartTest(TestCase):
    """购物车测试"""

    def setUp(self):
        self.client = Client()
        self.customer = User.objects.create_user(
            username="customer",
            password="testpass123",
            user_type=User.UserType.CUSTOMER,
        )
        self.wallet = Wallet.objects.create(
            user=self.customer,
            balance=Decimal("5000.00"),
        )
        self.seller = User.objects.create_user(
            username="seller",
            password="testpass123",
            user_type=User.UserType.SELLER,
        )
        self.product = Product.objects.create(
            name="测试商品",
            description="测试描述",
            price=Decimal("99.99"),
            stock=10,
            seller=self.seller,
            is_active=True,
        )

    def test_add_to_cart(self):
        """测试添加商品到购物车"""
        self.client.login(username="customer", password="testpass123")
        response = self.client.post(
            reverse("cart:add_to_cart", kwargs={"product_id": self.product.pk}),
            {"quantity": 2},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(CartItem.objects.count(), 1)
        cart_item = CartItem.objects.first()
        self.assertEqual(cart_item.quantity, 2)

    def test_add_same_product_again(self):
        """测试重复添加同一商品，数量累加"""
        self.client.login(username="customer", password="testpass123")
        self.client.post(
            reverse("cart:add_to_cart", kwargs={"product_id": self.product.pk}),
            {"quantity": 2},
        )
        self.client.post(
            reverse("cart:add_to_cart", kwargs={"product_id": self.product.pk}),
            {"quantity": 3},
        )
        cart_item = CartItem.objects.get(user=self.customer, product=self.product)
        self.assertEqual(cart_item.quantity, 5)

    def test_update_cart_item(self):
        """测试更新购物车商品数量"""
        self.client.login(username="customer", password="testpass123")
        cart_item = CartItem.objects.create(
            user=self.customer,
            product=self.product,
            quantity=2,
        )
        response = self.client.post(
            reverse("cart:update_cart", kwargs={"item_id": cart_item.pk}),
            {"quantity": 5},
        )
        self.assertEqual(response.status_code, 302)
        cart_item.refresh_from_db()
        self.assertEqual(cart_item.quantity, 5)

    def test_remove_from_cart(self):
        """测试从购物车移除商品"""
        self.client.login(username="customer", password="testpass123")
        cart_item = CartItem.objects.create(
            user=self.customer,
            product=self.product,
            quantity=2,
        )
        response = self.client.post(
            reverse("cart:remove_from_cart", kwargs={"item_id": cart_item.pk}),
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(CartItem.objects.count(), 0)

    def test_cart_total_price(self):
        """测试购物车总价计算"""
        CartItem.objects.create(
            user=self.customer,
            product=self.product,
            quantity=3,
        )
        cart_item = CartItem.objects.first()
        self.assertEqual(cart_item.get_total_price(), Decimal("299.97"))
