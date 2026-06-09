# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.urls import path
from . import views

app_name = "orders"

urlpatterns = [
    path("", views.order_list_view, name="order_list"),
    path("<int:pk>/", views.order_detail_view, name="order_detail"),
    path("buy/<int:product_id>/", views.buy_now, name="buy_now"),
    path("checkout/", views.checkout_cart, name="checkout"),
]
