# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.test import TestCase, Client
from django.urls import reverse
from decimal import Decimal
from accounts.models import User
from .models import Product


class ProductTest(TestCase):
    """商品测试"""

    def setUp(self):
        self.seller = User.objects.create_user(
            username="seller",
            password="testpass123",
            user_type=User.UserType.SELLER,
        )
        self.product = Product.objects.create(
            name="测试商品",
            description="这是一个测试商品",
            price=Decimal("99.99"),
            stock=10,
            seller=self.seller,
            is_active=True,
        )

    def test_product_creation(self):
        """测试商品创建"""
        self.assertEqual(self.product.name, "测试商品")
        self.assertEqual(self.product.price, Decimal("99.99"))
        self.assertEqual(self.product.stock, 10)
        self.assertTrue(self.product.is_active)

    def test_is_in_stock(self):
        """测试库存检查"""
        self.assertTrue(self.product.is_in_stock())

        self.product.stock = 0
        self.product.save()
        self.assertFalse(self.product.is_in_stock())

    def test_reduce_stock(self):
        """测试减少库存"""
        result = self.product.reduce_stock(3)
        self.assertTrue(result)
        self.assertEqual(self.product.stock, 7)

    def test_reduce_stock_insufficient(self):
        """测试库存不足"""
        result = self.product.reduce_stock(20)
        self.assertFalse(result)
        self.assertEqual(self.product.stock, 10)


class ProductViewTest(TestCase):
    """商品视图测试"""

    def setUp(self):
        self.client = Client()
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

    def test_product_list_view(self):
        """测试商品列表视图"""
        response = self.client.get(reverse("products:product_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "测试商品")

    def test_product_detail_view(self):
        """测试商品详情视图"""
        response = self.client.get(
            reverse("products:product_detail", kwargs={"pk": self.product.pk})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "测试商品")

    def test_inactive_product_not_shown(self):
        """测试下架商品不显示"""
        self.product.is_active = False
        self.product.save()
        response = self.client.get(reverse("products:product_list"))
        self.assertNotContains(response, "测试商品")
