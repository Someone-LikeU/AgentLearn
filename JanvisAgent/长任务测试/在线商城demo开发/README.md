# 在线商城系统

一个基于Django的在线商城系统demo，支持商家和客户两种角色，使用虚拟货币（软妹币）进行结算。

## 功能特性

### 用户系统
- 支持管理员、商家和客户三种角色注册/登录
- 客户注册后赠送5000软妹币
- 客户可在个人中心充值余额
- 登录失败5次后锁定30分钟，页面显示剩余尝试次数

### 商品系统
- 商家可创建新商品
- 商家可上架/下架商品
- 商品浏览和搜索

### 购物车系统
- 添加商品到购物车
- 购物车商品管理
- 批量结算

### 订单系统
- 立即购买
- 购物车结算
- 客户订单历史查看
- 商家可查看自己商品的订单

### 虚拟货币系统
- 软妹币账户管理
- 客户充值余额
- 商家销售需支付3%手续费

### 管理员系统
- 管理员登录后跳转到管理后台
- 查看商家列表（含商品名称、价格、销量、手续费）
- 查看客户列表（余额用***表示）

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 初始化数据库
```bash
cd mall_project
python manage.py makemigrations
python manage.py migrate
python manage.py init_data
```

### 运行开发服务器
```bash
python manage.py runserver
```

### 运行测试
```bash
python manage.py test
```

## 初始数据

### 管理员账号
| 用户名 | 密码 |
|--------|------|
| admin | admin123 |

### 商家账号
| 用户名 | 密码 |
|--------|------|
| seller1 | test123456 |
| seller2 | test123456 |
| seller3 | test123456 |

### 客户账号
| 用户名 | 密码 | 初始余额 |
|--------|------|----------|
| customer1 | test123456 | 5000 软妹币 |
| customer2 | test123456 | 5000 软妹币 |

### 初始商品
| 商品名称 | 价格 | 库存 | 商家 |
|----------|------|------|------|
| iPhone 15 Pro | 8999 软妹币 | 50 | seller1 |
| MacBook Air M3 | 9499 软妹币 | 30 | seller1 |
| AirPods Pro 2 | 1899 软妹币 | 100 | seller1 |
| 小米14 Ultra | 6499 软妹币 | 40 | seller2 |
| 华为MatePad Pro | 4999 软妹币 | 25 | seller2 |
| 索尼WH-1000XM5 | 2699 软妹币 | 60 | seller2 |
| 任天堂Switch OLED | 2599 软妹币 | 80 | seller2 |
| 戴森V15吸尘器 | 5490 软妹币 | 20 | seller3 |
| Kindle Paperwhite | 1068 软妹币 | 150 | seller3 |

## 技术栈
- Python 3.10+
- Django 4.x
- SQLite
- HTML5 + CSS3

## 项目结构
```
├── docs/                    # 项目文档
├── mall_project/           # 项目配置
├── accounts/               # 用户账户应用
├── products/               # 商品应用
├── cart/                   # 购物车应用
├── orders/                 # 订单应用
└── templates/              # 模板文件
```
