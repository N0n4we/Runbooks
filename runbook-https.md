# 自签 HTTPS + SSH 隧道访问（端到端加密）

> 本手册按 2026-06-07 实际方案编写：forge-neo 用自签证书在 `127.0.0.1:7860` 提供 HTTPS，
> 通过 SSH 本地端口转发从本机 `https://localhost:7860` 访问。

## 1. 架构

```
浏览器(localhost:7860)
   │  TLS(自签证书)  ← 在 forge 进程终结
   │  ── 整段 TLS 密文被塞进 ↓
SSH 隧道(-L 7860:127.0.0.1:7860，再加一层 SSH 加密)
   │
宿主机 :15864  ── L4 端口转发(NAT) ──→ 容器 172.17.0.4:22 (sshd)
                                              └─ 同容器 127.0.0.1:7860 (forge)
```
两层加密：SSH（客户端↔容器 sshd）+ TLS（浏览器↔forge 进程）。

## 2. 生成自签证书

```bash
cd ~/sd-webui-forge-neo && mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem -days 3650 \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```
- `-x509` 无 CA → 用自己的私钥给自己签名（`issuer == subject`），这就是"自签名"。
- `-nodes` 私钥不加密（服务启动免输密码）；`rsa:2048`；10 年有效。
- `CN=localhost` + SAN `localhost,127.0.0.1`：通过隧道以 `localhost` 访问时主机名能对上。
- 自签与正规证书**加密强度相同**，差别只是缺 CA 信任链 → 浏览器首次弹警告，手动信任即可。

## 3. 服务端启用（forge-neo）

在 `webui-user.sh` 的 `COMMANDLINE_ARGS` 中：
```bash
--tls-keyfile  $HOME/sd-webui-forge-neo/certs/key.pem
--tls-certfile $HOME/sd-webui-forge-neo/certs/cert.pem
# 不要加 --listen：不加时 gradio 绑定 127.0.0.1，只本机可达，正好配隧道
```
日志出现 `Running with TLS` 即生效。

## 4. 客户端访问

```bash
ssh -N -L 7860:127.0.0.1:7860 -p 15864 dxpb3F@wp08.unicorn.org.cn
# 浏览器打开 https://localhost:7860 ，自签证书点"高级/继续访问"
```
`-L 7860:127.0.0.1:7860` 里的 `127.0.0.1` 在 sshd 端（容器内）解析，直达同容器的 forge。

## 5. 端到端加密能否成立（含反向代理判断）

判端到端只看 **TLS 在哪里被解密**：本架构两端是「浏览器」和「forge 进程」（持有 `key.pem`）。

本次实测拓扑判断（容器内取证）：
```
SSH_CONNECTION = 172.17.0.1 38168 172.17.0.4 22   # 从网桥网关连到本容器 sshd:22
hostname       = a70fe84a8506                      # Docker 容器
监听            = 127.0.0.1:7860 (forge) + 0.0.0.0:22 (sshd)，本机 172.17.0.4
```
- 入口 `:15864` 前面**确实有一层转发**，但它是宿主机的 **L4 端口映射(NAT)**（源 IP 被改写成网关 `172.17.0.1`），只搬运密文，**不终结 SSH/TLS**。
- sshd 与 forge 在**同一容器**，隧道直达 forge。

**结论：端到端加密成立。** 只要中间层是 L4 转发、且不持有 forge 私钥，无论几跳代理都只能看到密文。
唯一能破坏的是「会终结 TLS 的七层代理」，但它必须换上自己的证书 → 指纹会变 → 可被发现。

## 6. 验证没有中间人（每次连上后可选做）

```bash
# 通过隧道看到的证书指纹，应与服务端一致
echo | openssl s_client -connect localhost:7860 2>/dev/null | openssl x509 -noout -fingerprint -sha256
```
本案例服务端证书指纹（基线，用于 TOFU/锁定比对）：
```
SHA256 = 12:1D:AD:52:D5:4E:62:9A:AC:86:92:2E:8F:45:99:4A:16:13:46:08:44:62:82:64:5C:36:82:FF:FA:57:A7:A6
subject = issuer = CN=localhost   (自签名特征)
有效期  = 2026-06-07 ~ 2036-06-04
```
两道可自校验的锁，即使完全不信任平台也能确证链路未被解开：
1. **SSH host key**（首次连接 TOFU）——若被 SSH 中间人终结，指纹变 → 客户端报警；
2. **TLS 证书指纹**（上面这串）——若被 TLS 中间人终结，浏览器看到的指纹变。

> 注：取证是在容器内进行的，可确认前面是 L4 NAT；无法从内部 100% 排除宿主机另置 MITM，
> 但只要上面两个指纹与首次一致，即证明"浏览器→forge 进程"全程未被任何中间层解密。
