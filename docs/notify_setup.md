# Stocka 自动化推送配置指南

本指南帮助你配置每日选股结果的自动推送功能，支持邮件、飞书、企业微信等渠道。

## 快速开始

### 1. 配置邮件推送（推荐）

编辑 `notify_config.yaml` 文件：

```yaml
email:
  enabled: true
  smtp_server: "smtp.qq.com"      # QQ邮箱服务器
  smtp_port: 465
  use_ssl: true
  username: "your_email@qq.com"   # 发件人邮箱
  password: "${EMAIL_PASSWORD}"   # 从环境变量读取
  from_addr: "your_email@qq.com"
  to_addrs:
    - "your_phone@example.com"    # 收件人邮箱
```

设置环境变量（密码/授权码）：

```bash
export EMAIL_PASSWORD="your_auth_code"
```

### 2. 测试配置

```bash
# 测试邮件发送
python -m quant_framework.cli notify test --channel email
```

### 3. 手动发送报告

```bash
# 发送最新的预测报告
python -m quant_framework.cli notify send --channel email

# 发送指定日期的报告
python -m quant_framework.cli notify send --date 2026-02-14
```

### 4. 配置定时任务

```bash
# 配置每日16:30自动运行
./scripts/setup_cron.sh

# 查看状态
./scripts/setup_cron.sh --status

# 移除定时任务
./scripts/setup_cron.sh --remove
```

## 配置文件详解

### 邮件配置 (SMTP)

常用邮箱服务器设置：

| 邮箱 | SMTP服务器 | 端口 | 说明 |
|------|-----------|------|------|
| QQ邮箱 | smtp.qq.com | 465 | 需开启SMTP，使用授权码 |
| 163邮箱 | smtp.163.com | 465 | 需开启SMTP，使用授权码 |
| Gmail | smtp.gmail.com | 587 | 需使用应用专用密码 |
| Outlook | smtp.office365.com | 587 | 使用登录密码 |

**获取授权码**：
- QQ邮箱：设置 → 账户 → 开启SMTP服务 → 生成授权码
- 163邮箱：设置 → POP3/SMTP/IMAP → 开启服务 → 获取授权码

### 飞书配置

1. 在飞书群中添加"自定义机器人"
2. 复制 Webhook 地址
3. 如需安全验证，勾选"签名校验"并复制密钥

```yaml
feishu:
  enabled: true
  webhook_url: "https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
  secret: "${FEISHU_SECRET}"  # 签名校验密钥
```

### 企业微信配置

1. 在群聊中添加"群机器人"
2. 选择"自定义"机器人
3. 复制 Webhook 地址中的 key 参数

```yaml
wechat:
  enabled: true
  webhook_url: "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx"
```

## 环境变量配置

为安全起见，敏感信息建议通过环境变量设置：

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export EMAIL_PASSWORD="your_email_auth_code"
export FEISHU_SECRET="your_feishu_secret"
export WECHAT_KEY="your_wechat_key"
```

或者创建 `.env` 文件（不要提交到git）：

```bash
cp .env.example .env
# 编辑 .env 文件填写实际值
```

## 云服务器部署

### 使用 systemd（推荐）

创建服务文件 `/etc/systemd/system/stocka-notify.service`：

```ini
[Unit]
Description=Stocka Daily Stock Pick Notification
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/stocka
Environment=EMAIL_PASSWORD=${EMAIL_PASSWORD}
ExecStart=/home/ubuntu/stocka/venv/bin/python -m quant_framework.scheduler.trading_scheduler
```

创建定时器 `/etc/systemd/system/stocka-notify.timer`：

```ini
[Unit]
Description=Run Stocka notification at 16:30 on trading days

[Timer]
OnCalendar=Mon-Fri 16:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable stocka-notify.timer
sudo systemctl start stocka-notify.timer
```

### 使用 Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

# 设置环境变量
ENV EMAIL_PASSWORD=${EMAIL_PASSWORD}

# 使用cron运行
RUN apt-get update && apt-get install -y cron
RUN echo "30 16 * * 1-5 cd /app && python -m quant_framework.scheduler.trading_scheduler >> /var/log/cron.log 2>&1" | crontab -
CMD ["cron", "-f"]
```

## 故障排查

### 测试邮件失败

1. 检查SMTP配置（服务器、端口、SSL设置）
2. 确认授权码正确（不是登录密码）
3. 查看邮箱是否开启SMTP服务
4. 检查是否开启"允许不够安全的应用访问"（Gmail）

### 定时任务未执行

```bash
# 查看cron日志
tail -f logs/cron.log

# 手动测试调度器
python -m quant_framework.scheduler.trading_scheduler --force

# 检查今天是否为交易日
python -m quant_framework.scheduler.trading_scheduler --check-trading-day
```

### 权限问题

```bash
# 确保日志目录可写
mkdir -p logs
chmod 755 logs

# 确保配置文件可读
chmod 600 notify_config.yaml
```

## 常见问题

**Q: 可以发送到多个邮箱吗？**
A: 可以，在 `to_addrs` 列表中添加多个邮箱地址。

**Q: 支持短信通知吗？**
A: 目前不支持直接短信，但可以通过以下方式间接实现：
- 邮箱绑定微信/QQ，新邮件会收到推送
- 使用支持短信的Webhook服务

**Q: 如何修改发送时间？**
A: 编辑 `notify_config.yaml` 中的 `scheduler.run_time`，然后重新运行 `setup_cron.sh`。

**Q: 非交易日会发送吗？**
A: 默认不会。设置 `scheduler.trading_days_only: false` 可改为每天发送。

## 安全建议

1. **不要将密码写入配置文件** - 使用环境变量或 `.env` 文件
2. **限制配置文件权限** - `chmod 600 notify_config.yaml`
3. **定期更换授权码** - 邮箱授权码建议定期更换
4. **使用专用邮箱** - 建议使用专门的邮箱发送通知

## 联系支持

如有问题，请查看：
- 日志文件：`logs/notify.log` 和 `logs/cron.log`
- 项目文档：`docs/` 目录
- 提交 Issue：项目仓库 Issues 页面
