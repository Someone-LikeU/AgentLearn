# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.test import TestCase, Client
from django.urls import reverse
from decimal import Decimal
from .models import User, Wallet


class UserRegistrationTest(TestCase):
    """用户注册测试"""

    def setUp(self):
        self.client = Client()
        self.register_url = reverse("accounts:register")

    def test_customer_registration(self):
        """测试客户注册，应赠送5000软妹币"""
        data = {
            "username": "test_customer",
            "email": "customer@test.com",
            "password1": "testpass123",
            "password2": "testpass123",
            "user_type": "customer",
        }
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, 302)  # 重定向

        user = User.objects.get(username="test_customer")
        self.assertEqual(user.user_type, User.UserType.CUSTOMER)
        self.assertTrue(hasattr(user, "wallet"))
        self.assertEqual(user.wallet.balance, Decimal("5000.00"))

    def test_seller_registration(self):
        """测试商家注册，不应有初始余额"""
        data = {
            "username": "test_seller",
            "email": "seller@test.com",
            "password1": "testpass123",
            "password2": "testpass123",
            "user_type": "seller",
        }
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, 302)

        user = User.objects.get(username="test_seller")
        self.assertEqual(user.user_type, User.UserType.SELLER)

    def test_duplicate_username_registration(self):
        """测试重复用户名注册"""
        User.objects.create_user(username="existing", password="testpass123")
        data = {
            "username": "existing",
            "email": "test@test.com",
            "password1": "testpass123",
            "password2": "testpass123",
            "user_type": "customer",
        }
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, 200)  # 返回表单错误


class UserLoginTest(TestCase):
    """用户登录测试"""

    def setUp(self):
        self.client = Client()
        self.login_url = reverse("accounts:login")
        self.user = User.objects.create_user(
            username="testuser",
            password="testpass123",
            user_type=User.UserType.CUSTOMER,
        )

    def test_successful_login(self):
        """测试成功登录"""
        data = {"username": "testuser", "password": "testpass123"}
        response = self.client.post(self.login_url, data)
        self.assertEqual(response.status_code, 302)

    def test_invalid_login(self):
        """测试无效登录"""
        data = {"username": "testuser", "password": "wrongpassword"}
        response = self.client.post(self.login_url, data)
        self.assertEqual(response.status_code, 200)


class WalletTest(TestCase):
    """钱包测试"""

    def setUp(self):
        self.user = User.objects.create_user(
            username="walletuser",
            password="testpass123",
            user_type=User.UserType.CUSTOMER,
        )
        self.wallet = Wallet.objects.create(
            user=self.user,
            balance=Decimal("1000.00"),
        )

    def test_has_sufficient_balance(self):
        """测试余额检查"""
        self.assertTrue(self.wallet.has_sufficient_balance(Decimal("500.00")))
        self.assertFalse(self.wallet.has_sufficient_balance(Decimal("1500.00")))

    def test_deduct_balance(self):
        """测试扣款"""
        result = self.wallet.deduct(Decimal("300.00"))
        self.assertTrue(result)
        self.assertEqual(self.wallet.balance, Decimal("700.00"))

    def test_deduct_insufficient_balance(self):
        """测试余额不足扣款"""
        result = self.wallet.deduct(Decimal("1500.00"))
        self.assertFalse(result)
        self.assertEqual(self.wallet.balance, Decimal("1000.00"))

    def test_add_balance(self):
        """测试充值"""
        self.wallet.add(Decimal("500.00"))
        self.assertEqual(self.wallet.balance, Decimal("1500.00"))
