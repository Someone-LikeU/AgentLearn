# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.urls import path
from . import views

app_name = "accounts"

urlpatterns = [
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("profile/", views.profile_view, name="profile"),
    path("admin-dashboard/", views.admin_dashboard, name="admin_dashboard"),
]
