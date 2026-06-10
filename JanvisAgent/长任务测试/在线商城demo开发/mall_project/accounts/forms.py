# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.utils.safestring import mark_safe
from .models import User


class HorizontalRadioSelect(forms.RadioSelect):
    """横向排列的单选按钮"""

    def render(self, name, value, attrs=None, renderer=None):
        """直接返回横向排列的HTML，不依赖外部模板文件"""
        choices = list(self.choices)
        html_parts = []
        for option_value, option_label in choices:
            checked = 'checked' if str(option_value) == str(value) else ''
            html_parts.append(
                f'<label style="display:inline-flex;align-items:center;gap:4px;'
                f'margin-right:15px;cursor:pointer;font-weight:normal;">'
                f'<input type="radio" name="{name}" value="{option_value}" {checked}>'
                f'{option_label}</label>'
            )
        return mark_safe(''.join(html_parts))


class UserRegisterForm(UserCreationForm):
    """用户注册表单"""

    USER_TYPE_CHOICES = [
        ("customer", "客户"),
        ("seller", "商家"),
    ]

    user_type = forms.ChoiceField(
        choices=USER_TYPE_CHOICES,
        widget=HorizontalRadioSelect,
        label="用户类型",
    )
    email = forms.EmailField(required=True, label="邮箱")

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2", "user_type")
        labels = {
            "username": "用户名",
            "password1": "密码",
            "password2": "确认密码",
        }


class UserLoginForm(AuthenticationForm):
    """用户登录表单"""

    username = forms.CharField(label="用户名")
    password = forms.CharField(label="密码", widget=forms.PasswordInput)
