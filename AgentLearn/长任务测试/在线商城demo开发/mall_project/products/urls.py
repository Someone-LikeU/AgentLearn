# encoding: utf-8
# @Time    : 2026/06/09 11:46

from django.urls import path
from . import views

app_name = "products"

urlpatterns = [
    path("", views.product_list_view, name="product_list"),
    path("<int:pk>/", views.product_detail_view, name="product_detail"),
    path("my/", views.my_products_view, name="my_products"),
    path("create/", views.create_product, name="create_product"),
    path("<int:pk>/toggle/", views.toggle_product_status, name="toggle_status"),
]
