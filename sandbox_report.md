# Agent Sandbox：运行时后端、接口与 RL Infra 集成调研

> 调研截止：2026-08-11
> 范围：学术论文、官方技术文档、开源仓库的主分支、release、issue 与 pull request。性能数字均标注其性质；除非明确说明，它们不是同一硬件、同一镜像、同一负载下的横向 benchmark。

## 摘要

Agent Sandbox 不是某一种虚拟化技术，而是一个完整的执行系统：它要把模型生成的、默认不可信的动作放入受控环境，同时向 agent 提供 shell、文件、网络、浏览器、包管理、长进程、端口和状态持久化等能力。**runtime backend 决定了最重要的隔离边界与性能下限，但最终安全性和可用性还取决于镜像供应链、挂载、网络出口、secret 注入、资源配额、生命周期控制和审计。** “用了 VM”并不自动安全；把宿主工作区读写挂进去、允许任意 egress 或把长期凭据放进 guest，仍然可能造成严重损失。

六类后端可以用一个连续谱理解：

`进程/namespace（bubblewrap） → 共享内核容器（Docker/runc） → 用户态内核（gVisor） → 精简硬件 VM（microVM） → 完整 VM → 能力导入受限的语言 VM（Wasm）`

其中 Wasm 不是这条谱系的简单“更轻端”：它改变了 guest ABI，牺牲完整 Linux 兼容性，换取极低启动成本、可移植性和可审计的 capability surface。对于任意仓库、任意二进制、浏览器和 Docker-in-Docker，microVM 通常是当前云端多租户 Agent Sandbox 的主流强隔离选择；对于本地 coding agent，在用户已经信任当前仓库、且低延迟和原生工具兼容优先时，bubblewrap/Seatbelt 更常见；Docker 仍是评测、CI 和可信集群中部署成本最低的基线；gVisor 是 OCI 兼容与更强内核隔离之间的折中。

在 RL Infra 中，Sandbox 也不应只被视为 `exec()` RPC。一次 rollout session 至少涉及：分配/恢复环境、准备镜像与任务数据、执行多轮动作、采集精确 token 与环境观测、心跳和超时、快照/分叉、验证、回收及失败归因。当前没有统一的 Agent Sandbox 标准协议；行业正在形成若干不同层次的“局部标准”：OCI/CRI、Kubernetes Agent Sandbox CRD、E2B-compatible SDK、MCP，以及 Gym/GEM 风格 `reset/step/close`。

---

## 1. 讨论 Agent Sandbox 时应关注哪些因素

以下是当前论文、项目设计和工程评估中反复出现的维度。它们是一份 checklist，不在本节逐项展开：

1. **隔离与攻击面**：隔离边界是否跨内核；host syscall 暴露面；VMM/设备模型大小；guest-to-host escape 难度；多租户互相隔离。
2. **权限与策略**：filesystem mount、Linux capability、seccomp、root/rootless、网络 allow/deny、DNS、credential injection、人工 approval。
3. **资源隔离与抗滥用**：CPU、内存、PIDs、磁盘容量/IOPS、网络带宽、fork bomb、超时、GPU 和设备配额。
4. **启动与交互性能**：冷启动、热启动、单次 `exec` 延迟、文件 I/O、网络 I/O、系统调用密集负载、并发创建吞吐。
5. **密度与成本**：每实例固定内存、宿主可承载实例数、镜像和快照存储、跨节点传输、空闲实例成本。
6. **Linux/POSIX 兼容性**：能否运行任意 ELF、动态链接、包管理器、浏览器、FUSE、Docker、内核特性、长进程和 PTY。
7. **可移植性与硬件要求**：Linux/macOS/Windows、KVM/嵌套虚拟化、ARM64、Kubernetes、HPC/rootless 环境。
8. **功能完整性**：命令、文件、上传下载、流式 stdout/stderr、PTY、端口暴露、浏览器/桌面、MCP、GPU、Docker-in-Docker。
9. **状态与生命周期**：create/connect/kill、idle timeout、pause/resume、snapshot/clone/rollback、持久卷、跨节点恢复、版本兼容。
10. **镜像与环境复现**：OCI 兼容、镜像按需加载、预热池、依赖缓存、确定性、SBOM、漏洞修复和基础镜像更新。
11. **网络与 secret 安全**：egress policy、TLS 代理、metadata service/SSRF、secret 是否进入 guest、域名重绑定、审计与撤销。
12. **可观测性与调试**：命令/文件/网络审计、metrics、trace、录像、资源统计、失败原因、可重放性。
13. **控制面可靠性**：调度、配额、租户身份、幂等、租约、孤儿回收、控制面重启、节点故障、resume storm。
14. **开发与运维复杂度**：API/SDK 成熟度、部署依赖、升级路径、内核/VMM 维护、Kubernetes 集成、社区活跃度。
15. **威胁模型与合规**：代码是否来自陌生租户；数据是否敏感；是否需要强租户边界、取证、地域约束和保留策略。
16. **Agent/RL 特有语义**：session affinity、并行 rollout、快慢样本、环境分叉、干净 verifier、reward 防作弊、policy staleness。

这里最容易犯的错误是只比较“启动多少毫秒”。对 agent 而言，依赖安装、仓库扫描、浏览器启动和模型推理常常比 VMM boot 更长；而对大规模 RL，P99 创建延迟、快照存储放大、节点缓存命中率、失败恢复和调度背压往往比单实例 P50 更重要。

---

## 2. 六类 runtime backend 的宏观分类

### 2.1 核心比较表

| 后端 | 主要隔离边界 | 性能/密度特征 | 兼容性与功能 | 主要优势与适合场景 | 主要缺点或风险 |
|---|---|---|---|---|---|
| **bubblewrap** | 同一 host kernel；为进程创建 user/mount/PID/network 等 namespace，构造最小 rootfs，并由调用者叠加 seccomp/cgroup/LSM | 几乎没有 VM/daemon 冷启动；CPU 接近原生；内存固定开销很低 | 原生 Linux userspace，宿主已有工具可直接使用；镜像、网络、cgroup、审计均需上层自己组合 | 本地 coding agent、桌面应用、单用户且需要毫秒级 shell；短命令、高频工具调用 | 内核漏洞仍跨越隔离边界；策略完全取决于参数和挂载；不当 bind mount、D-Bus、TTY、网络代理可破坏边界；不是多租户编排系统。bubblewrap README 也明确说“安全模型由调用参数决定”并警告所有挂载资源都可能成为逃逸通道（[项目说明](https://github.com/containers/bubblewrap)） |
| **Docker/runc 容器** | 同一 host kernel；namespace + cgroup + capability + seccomp + AppArmor/SELinux | 秒以下启动、镜像生态成熟、密度高；文件层和 daemon 有额外开销 | 高 Linux 兼容；OCI 镜像、volume、network、GPU、K8s 生态最好 | CI/评测、可信或同组织 workload、快速搭建 RL 环境、开发基线 | 共享内核；`--privileged`、Docker socket、危险 mount 会近乎取消隔离；daemon/镜像供应链和默认 egress 是额外风险。Docker 默认 seccomp 只是一层纵深防御（[seccomp 文档](https://docs.docker.com/engine/security/seccomp/)） |
| **gVisor/runsc** | guest syscall 先进入用户态内核 **Sentry**；文件访问常经 **Gofer**；宿主仅暴露受限 syscall，最终仍使用 host kernel | CPU 计算接近容器；syscall、文件和网络 I/O 可能显著变慢；固定内存高于 runc、通常低于 VM | OCI/Kubernetes 兼容较好，但不是完整 Linux；部分 syscall、设备、性能工具、GPU 能力有限 | 希望沿用容器交付和 K8s 运维，同时降低直接攻击 host kernel 的风险；多租户 code execution 的中间档 | Sentry 本身是较大的用户态内核攻击面；兼容性边角和 I/O 性能成本；需要按 workload benchmark。其性能文档明确指出额外 syscall 层和 filesystem/network 是主要成本（[架构](https://gvisor.dev/docs/architecture_guide/intro/)、[性能](https://gvisor.dev/docs/architecture_guide/performance/)） |
| **microVM** | KVM/硬件虚拟化 + 独立 guest kernel；VMM 只实现精简设备模型；通常再对 VMM 进程做 jail/seccomp/cgroup | 冷启动可到几十至百毫秒级；固定内存通常数 MiB 级 VMM 开销加 guest 内存；密度低于容器、高于传统 VM；snapshot 可把恢复降到毫秒级 | 完整 Linux guest，root、包管理、任意 ELF、浏览器、嵌套容器较容易；设备面比完整 VM 少 | 公有云、多租户不可信代码、长时间自主 agent、RL rollout farm、需要 snapshot/fork | 依赖 KVM/嵌套虚拟化；控制面、网络、镜像、快照和 guest agent 更复杂；VMM/KVM 仍可能有漏洞；GPU、热插拔、迁移等能力取决于 VMM。Firecracker 自身不做 egress 过滤，必须由 host 层实现（[设计](https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md)） |
| **完整 VM** | hypervisor + 完整 guest kernel/OS/设备模型；每租户/实例独立机器语义 | 启动通常秒到分钟；内存/磁盘开销最大，密度最低；成熟镜像/休眠/迁移可改善 | 兼容性最强，可运行不同 OS、内核模块、复杂网络、桌面和设备 | 高价值长会话、完整 computer-use、Windows/macOS/Linux 环境、需要强 OS 语义或特殊设备 | 成本、调度粒度、镜像体积、补丁和启动时间；QEMU 等完整设备模拟扩大 VMM 攻击面（[QEMU 安全文档](https://www.qemu.org/docs/master/system/security.html)） |
| **Wasm/WASI** | 模块只能访问线性内存和显式 import；host 通过 WASI/capability 暴露文件、时钟、网络等 | 微秒到毫秒级实例化潜力、密度高、可预编译；跨边界调用和 JIT/AOT 有成本 | 不是完整 Linux ABI；只能运行编译到 Wasm 的程序或受控解释器，包管理器、浏览器、任意 ELF、daemon/GPU 较弱 | 纯计算工具、插件、函数、数据变换、确定性评测、浏览器内 agent tool；希望最小 capability surface | 生态和 POSIX 兼容不足；runtime/JIT bug 仍可逃逸；WASI capability 不等于 CPU/内存/线程/磁盘配额，资源耗尽仍需 host 控制（[Wasm 安全模型](https://webassembly.org/docs/security/)、[Wasmtime 安全](https://docs.wasmtime.dev/security.html)） |

### 2.2 核心区别，不应混为一谈的三组概念

**bubblewrap 与 Docker** 都共享宿主内核，但抽象层次不同。bubblewrap 是“为一个进程构造 namespace/mount 视图”的低层工具，没有镜像规范、daemon、网络模型和集群控制面；Docker 是围绕 OCI 镜像、生命周期、存储和网络形成的容器平台。相同内核边界不意味着工程能力相同。

**gVisor 与 microVM** 都减少 guest 直接触碰 host kernel 的机会，但方式不同。gVisor 在用户态重新实现 Linux syscall 语义，仍是 OCI container UX；microVM 让 guest syscall 进入自己的 guest kernel，host 看到的是 KVM exit 和虚拟设备 I/O。前者通常部署更轻，后者兼容性和硬件级边界更清晰。

**microVM 与完整 VM** 共享硬件虚拟化边界。区别主要是设备模型、启动路径和产品目标，而不是“有没有真正的 VM”：microVM 删去 BIOS、legacy device 和通用硬件模拟，针对同架构 Linux workload；完整 VM 优先兼容性、设备和多 OS。Firecracker 论文报告的 `<125 ms` 启动和 `<5 MiB` VMM 内存是其特定配置的结果，不应直接外推到带 guest agent、网络、镜像恢复和业务初始化的整套 Sandbox（[NSDI 2020 论文](https://www.usenix.org/conference/nsdi20/presentation/agache)）。

---

## 3. 每类后端内部的实现路径与差异

### 3.1 bubblewrap / OS-native sandbox

| 实现 | 机制 | 差异对应的因素 |
|---|---|---|
| bubblewrap | Linux user/mount/PID/network namespace，按参数构造只读/读写 bind mount；常与 seccomp、cgroup、代理组合 | 最低启动成本和原生兼容；策略正确性、Linux-only、共享内核风险 |
| macOS Seatbelt / `sandbox-exec` | macOS Sandbox profile 约束文件和网络等操作 | 本地 UX 好，但私有/演进中的 OS policy 语义、跨平台一致性差 |
| Landlock/seccomp 等内核机制 | 直接在进程层限制路径和 syscall，通常作为纵深防御而非完整产品 | policy 粒度、可组合性和内核版本；本身不提供镜像与生命周期 |

Anthropic 的开源 [sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime) 展示了典型产品化组合：Linux 用 bubblewrap，macOS 用 Seatbelt，再加 host proxy 做域名级 egress。它的价值不是发明新隔离边界，而是把跨平台策略、网络代理和 coding-agent UX 组合起来。

### 3.2 Docker/runc 容器

容器内部的主要实现差异不在 namespace 原理，而在以下因素：rootless 或 rootful；runc/crun；seccomp/LSM 默认策略；overlayfs/storage driver；Docker daemon、containerd/CRI-O；单容器单 session 或常驻 worker 内 `docker exec`；是否预热镜像；是否给 guest 挂 Docker socket。对 Agent Sandbox 而言，**“Docker backend”必须继续说明是否 privileged、挂载了什么、网络是否 deny-by-default、一个容器是否复用多个租户**。

### 3.3 gVisor

gVisor 的 OCI runtime 是 `runsc`，主要 platform 路径包括：

- **systrap**：利用 host syscall/进程机制截获 guest syscall，是目前常用默认路径；不要求 `/dev/kvm`，在嵌套环境更易部署。
- **KVM platform**：用 KVM 提供地址空间切换/截获机制，但它仍不是“给 workload 一个完整 guest kernel”的 microVM；Linux 语义仍由 Sentry 实现。
- **Gofer/直接文件路径、network stack、nvproxy**：决定 I/O 开销、host surface 和 GPU 兼容。GPU 通过 [nvproxy](https://gvisor.dev/docs/user_guide/gpu/) 暴露经验证的驱动接口，而不是无条件设备透传。

因此 gVisor 项目间的性能差异常常来自 platform、filesystem access type、网络栈和 workload syscall profile，而不是一个统一的“gVisor 慢多少”。

### 3.4 microVM：Firecracker、Cloud Hypervisor、Kata 的关系

首先要纠正一个常见分类错误：**Kata Containers 不是与 Firecracker/Cloud Hypervisor 并列的 VMM。** Kata 是 OCI/CRI 兼容的 secure-container runtime 与编排层，底下可选择 QEMU、Cloud Hypervisor、Firecracker、Dragonball 等 VMM；它在 guest 内再启动 container runtime/agent，把“Pod/Container”映射到轻量 VM（[Kata 架构](https://github.com/kata-containers/kata-containers/blob/main/docs/design/architecture/README.md)、[VMM 支持矩阵](https://github.com/kata-containers/kata-containers/blob/main/docs/design/virtualization.md)）。

| 实现 | 设计中心 | 优势 | 代价/限制 | 主要关联因素 |
|---|---|---|---|---|
| **Firecracker** | serverless、多租户、每进程一个极简 microVM；virtio block/net/vsock，REST API，配套 jailer | 设备面小、冷启动和密度优秀、snapshot 成熟、云上实践最多 | Linux/KVM、同架构；设备/热插拔/迁移能力刻意受限；网络和镜像控制面需自建 | 安全攻击面、启动、密度、snapshot |
| **Cloud Hypervisor** | Rust VMM，面向现代 cloud workload；复用 rust-vmm，强调 virtio、热插拔、设备透传、live migration | 比 Firecracker 功能更广，仍比传统 QEMU 精简；适合通用云 VM/Kata | 功能越广设备面和运维越复杂；启动/密度未必优于极简 Firecracker | 功能、设备/GPU、迁移、兼容性（[官方介绍](https://www.cloudhypervisor.org/docs/prologue/introduction/)） |
| **QEMU microVM machine type** | 在成熟 QEMU 中使用精简 machine/device 配置 | 兼容和工具最成熟，Kata 支持面广 | 二进制与设备模拟面较大；极致启动/内存通常不如专用 VMM | 成熟度、兼容性、攻击面 |
| **Kata Containers** | 用 VM 承载 OCI Pod/Container，接入 containerd/CRI/Kubernetes | 不改应用交付方式即可获得独立 kernel；K8s 运维体验好 | guest agent、shim、VMM 多层组件；启动与排错链更长 | 通用性、K8s 集成、运维复杂度 |
| **CubeHypervisor** | CubeSandbox 的 rust-vmm/KVM 定制 VMM，与 CubeCoW、CubeVS、CubeEgress 协同 | 为 sandbox create/snapshot/clone 和网络策略做端到端优化 | 相对新，生态与独立复用性不如 Firecracker；跨节点恢复尚在演进 | snapshot、网络、控制面、成熟度 |

### 3.5 完整 VM

完整 VM 的实现差异主要落在 hypervisor/VMM（KVM+QEMU、Hyper-V、VMware、Apple Virtualization Framework）、guest OS、设备直通、休眠/迁移、镜像格式和云控制面。对 agent 更实际的选择通常不是单独挑 QEMU，而是挑 AWS/Azure/GCP VM、开发机 VM 或完整 desktop-as-a-service。它以更高成本换取 Windows/完整桌面/内核模块/复杂企业网络等 microVM 有意删掉的能力。

### 3.6 Wasm/WASI

Wasm 方案内部差异包括：

- **Wasmtime/Cranelift**：WASI 和 Component Model 推进快，Rust 生态和安全流程成熟；可用 fuel/epoch interruption、memory limits 等 host controls。
- **Wasmer/Wasmer Edge、WasmEdge**：分别侧重多后端/边缘和 cloud-native/AI 插件生态。
- **WASI Preview 1/2/3、Component Model**：决定 socket、async、resource handle 和组件组合能力；能力模型仍在演进。
- **“完整程序编译到 Wasm”与“在 Wasm 中运行解释器”**：后者如 CPython/QuickJS 提升功能，但增加体积和攻击面，性能也不同。

[Vercel just-bash](https://github.com/vercel-labs/just-bash) 是很有代表性的 agent tool：它用 TypeScript 虚拟 shell/文件系统，Python 和 QuickJS 作为可选 Wasm 能力，网络默认关闭并按 URL/method allowlist 开放。项目自己明确建议：开发和测试可用它，任意二进制或完整 VM 需求应切到真正 Sandbox。因此它说明了 Wasm 的正确定位——**受控工具执行面**，而非所有 coding-agent workload 的透明替代。

---

## 4. 近期项目、各自优势与 harness 的实际选择

### 4.1 按 backend 列举代表性实践

| 后端 | 近期代表项目 | 核心优势/差异 | 性能判断 |
|---|---|---|---|
| bubblewrap / Seatbelt | Codex CLI、本地 Claude Code、Anthropic sandbox-runtime、Bubblewrap 本身 | Codex/Claude 把 OS sandbox 嵌入本地 approval/权限 UX；sandbox-runtime 更适合被其他程序复用 | 命令启动和 CPU 接近原生；主要成本来自代理、文件扫描和实际工具，不是 VM boot |
| Docker/runc | [agent-infra/sandbox](https://github.com/agent-infra/sandbox)、[SandboxFusion](https://github.com/bytedance/SandboxFusion)、[ROCK](https://github.com/alibaba/ROCK)、Uni-Agent Docker backend、OpenSandbox 默认本地 backend | agent-infra 把 Browser/Shell/File/MCP/VSCode/Jupyter 放进一个 AIO 容器；SandboxFusion 面向多语言代码评测；ROCK 提供分布式 env/sandbox 管理；Uni-Agent 统一 provider API | 镜像已缓存时创建快、吞吐高；AIO 镜像大，browser/Jupyter 初始化可能主导；安全上仍是共享 kernel |
| gVisor | [Kubernetes SIG Agent Sandbox](https://github.com/kubernetes-sigs/agent-sandbox)、GKE Agent Sandbox、OpenSandbox `runsc` backend | K8s Agent Sandbox 用标准 `runtimeClassName: gvisor` 叠加 Sandbox/SandboxClaim/WarmPool 生命周期，不把 runtime 写死；OpenSandbox 让同一 API 切换 runc/runsc/Kata | 比 runc 多 syscall/I/O 成本，预热池可隐藏创建延迟；应按编译、git、浏览器等真实 workload 测量 |
| microVM | [E2B](https://github.com/e2b-dev/infra)、[AgentENV](https://github.com/kvcache-ai/AgentENV)、[CubeSandbox](https://github.com/TencentCloud/CubeSandbox)、[Vercel Sandbox](https://vercel.com/docs/sandbox)、Docker Sandboxes、Firecracker、Kata | E2B 建立成熟云 API/模板/快照生态；AgentENV 强在 OverlayBD+ublk 的镜像/快照数据面；Cube 强在完整控制面、eBPF 网络/egress/credential；Vercel 强在托管开发者体验；Docker Sandboxes 强在本地 agent 内可安全运行独立 Docker daemon | Firecracker 原语可 `<125 ms`；具体产品常称几十毫秒，但镜像、网络、guest agent 与缓存条件不同。AgentENV/Cube 的数字见第 7 节，不做直接排名 |
| 完整 VM | 云 VM、自托管 QEMU/KVM、[OpenComputer](https://www.getoc.ai/) 等完整 computer-use 环境 | 支持完整 OS/桌面/企业软件和长任务，容易为每任务提供真正机器语义 | 冷启动、内存和存储最重；池化、休眠和镜像克隆可缓解，适合低并发高价值任务 |
| Wasm/受控语言 VM | Wasmtime/WASI、WasmEdge、just-bash、Extism 类插件系统 | capability surface 小、可嵌入、跨平台、适合把单个 tool 变成安全函数 | 对支持的纯计算/短工具启动极快；不能把任意 Linux agent workload 的 benchmark 与之直接比较 |

补充两类值得关注的“跨 backend 控制面”：

- [OpenSandbox](https://github.com/alibaba/OpenSandbox) 同时支持本地 Docker/Kubernetes，以及 gVisor、Kata/Firecracker 等 secure runtime；核心价值是 API 和调度解耦，而不是新的隔离原语。
- [Kubernetes SIG Agent Sandbox](https://agent-sandbox.sigs.k8s.io/) 把 Sandbox、Template、Claim、WarmPool 做成 CRD。2026 年 v0.5.0 将 API 升到 `v1beta1`，用 `spec.operatingMode: Running|Suspended` 表达暂停/恢复，并加入 gVisor、AKS Kata、RayJob 和 MCP 示例；release 同时修复了 redirect/SSRF、保留 metadata 被租户覆盖等安全问题（[release notes](https://github.com/kubernetes-sigs/agent-sandbox/releases)）。这说明控制面 API、router 和 SDK 同样属于威胁模型。

### 4.2 Claude Code、Codex、Pi 到底采用什么

| Harness | 默认/内置方案 | 结论与边界 |
|---|---|---|
| **Claude Code** | macOS 使用 Seatbelt，Linux/WSL2 使用 bubblewrap；网络通过 host proxy 控制 | 是本地 OS-native process sandbox，不是 Docker/microVM。若 sandbox 不可用，默认会告警并可能退回 unsandboxed；高保证部署应启用 `failIfUnavailable`。官方说明见 [Claude Code sandboxing](https://code.claude.com/docs/en/sandboxing) 和 [工程博客](https://www.anthropic.com/engineering/claude-code-sandboxing) |
| **Codex CLI 本地模式** | macOS Seatbelt；Linux/WSL2 bubblewrap；Windows 使用 Windows-native sandbox | 同样是本地 OS-native 方案，并与 approval policy 分开：sandbox 是技术边界，approval 是何时请求用户授权。见 [Codex sandboxing 文档](https://learn.chatgpt.com/docs/sandboxing)。Codex Cloud 的底层多租户 runtime 未在该文档中公开，不能据本地实现推断为同一 backend |
| **Pi** | 核心默认没有内置 sandbox | [Pi 安全文档](https://pi.dev/docs/latest/security) 把本地 agent 明确放在用户安全边界内；可由扩展或外层环境隔离，例如 Docker Sandbox、CubeSandbox 的 [Pi 集成](https://docs.cubesandbox.com/guide/integrations/pi-agent.html)。因此“Pi 使用 microVM”只对具体部署成立，不是 harness 默认 |

一个实用结论是：**harness 与 sandbox provider 正在解耦。** Claude Code/Codex/Pi 都能被放进 Docker Sandboxes 或云端 microVM；此时“内部 command policy”和“外部 machine boundary”可以叠加。Docker 2026 年的 `sbx` 就支持在独立 microVM 内运行 Claude/Codex 等 agent，并给每个 sandbox 独立 Docker daemon（[架构](https://docs.docker.com/ai/sandboxes/architecture/)、[安全模型](https://docs.docker.com/ai/sandboxes/security/)）。默认直挂工作区仍可让 agent 修改宿主代码；强边界应使用只读源仓库加私有 clone，而不是只看 hypervisor。

---

## 5. 项目接口如何设计，是否存在通用协议

### 5.1 结论：没有单一通用协议，只有分层兼容面

| 层 | 代表规范/API | 解决什么 | 不解决什么 |
|---|---|---|---|
| runtime/镜像 | OCI Image + Runtime Spec、containerd shim、Kubernetes CRI | image、bundle、process/container lifecycle | agent session、文件 RPC、snapshot/fork、reward |
| K8s 控制面 | Agent Sandbox `Sandbox`/`SandboxTemplate`/`SandboxClaim`/`SandboxWarmPool` CRD | 声明式分配、预热、暂停、路由、租约 | 统一所有 provider 的命令和快照语义 |
| Agent Sandbox SDK | E2B API/SDK 及大量 “E2B-compatible” 实现；OpenSandbox OpenAPI | create/connect/kill、command、file、port、timeout、pause/resume/snapshot | 不是标准组织批准的协议；错误码、PTY、snapshot 一致性和资源模型仍有 provider 差异 |
| 工具协议 | MCP | tool discovery、JSON schema、tool call/result、resource/prompt | 环境分配、强隔离、状态快照、RL trajectory |
| RL 环境 | Gym/GEM、AEnvironment 的扩展 MCP | `make/reset/step/close`、observation/action/reward/termination | 底层用 Docker、VM 或远程服务并不由它规定 |
| 模型网关 | OpenAI-compatible Chat Completions、Anthropic Messages | rollout 请求、token/logprob/路由 | sandbox 文件和机器生命周期 |

所以最合理的架构不是等待一个“大一统协议”，而是明确分层：**control plane**（分配、租约、pause、kill）、**data plane**（exec、file、PTY、ports）、**policy plane**（network、mount、secret、quota）和 **RL semantic plane**（reset、step、reward、trajectory）。

### 5.2 事实兼容面通常包含哪些接口

从 E2B、OpenSandbox、ROCK、Uni-Agent 和 Cube/AgentENV 的交集看，最小实用接口如下：

```text
Lifecycle:
  create(template|image, resources, env, policy, timeout) -> sandbox_id
  connect(sandbox_id, lease/token)
  alive/status/stats/set_timeout
  pause / resume / snapshot / clone / rollback       # provider-dependent
  kill/delete

Execution:
  exec(argv|shell, cwd, env, timeout) -> exit_code, stdout, stderr
  exec_stream(...) -> stdout/stderr/events
  create_pty / write_stdin / resize / signal / wait
  start_process / list_processes / kill_process

Filesystem:
  read/write/list/stat/mkdir/remove
  upload/download/archive

Connectivity:
  expose_port -> authenticated URL
  network policy / egress allowlist / credential binding

Observability:
  resource metrics, audit log, command event, network event, failure reason
```

E2B 把 `pause/resume` 定义为“一对一保留同一个 sandbox”，把 snapshot/template 定义为“一对多创建后继环境”，这一语义区分对 RL 特别重要（[E2B snapshot](https://e2b.dev/docs/sandbox/snapshots)、[persistence](https://e2b.dev/docs/sandbox/persistence)）。[OpenSandbox 架构](https://github.com/alibaba/OpenSandbox/blob/main/docs/architecture.md) 则用 OpenAPI 划分 lifecycle、execution、diagnostics 和 egress API。ROCK 进一步暴露 `commit image`、持久 bash session，以及 GEM 风格环境接口（[ROCK API](https://alibaba.github.io/ROCK/docs/1.1.x/References/api/)）。

### 5.3 一个推荐的 session 接口用法

```python
lease = await sandbox.create(
    image=task.image,
    resources={"cpu": 2, "memory_mb": 4096, "disk_mb": 20480},
    policy={"egress": task.allowlist, "secrets": task.secret_bindings},
    idempotency_key=rollout_id,
)
try:
    await sandbox.upload(lease.id, task.bundle, "/workspace")
    await sandbox.exec(lease.id, ["bash", "-lc", task.setup], timeout=300)
    checkpoint = await sandbox.snapshot(lease.id)       # optional
    for turn in trajectory:
        result = await sandbox.exec_stream(lease.id, turn.command)
        await trajectory_log.append(result)
        await sandbox.heartbeat(lease.id)
    verdict = await verifier.run_from_clean_clone(checkpoint)
finally:
    await sandbox.kill(lease.id, reason="rollout-finished")
```

生产接口还应有：idempotency key、租约/心跳、可取消请求、结构化失败类别、最大输出截断、幂等 kill、provider capability discovery、image/kernel/snapshot 版本、审计关联 ID。否则训练端无法分清“策略做错”“命令超时”“sandbox 死亡”“基础设施重试”，会污染 reward 和有效样本统计。

---

## 6. RL Infra 如何集成 Sandbox

### 6.1 一个 rollout session 的典型交互过程

| 生命周期阶段 | RL Infra ↔ Sandbox 关键接口 | 为什么重要 |
|---|---|---|
| 0. 能力发现与排队 | `capabilities`, image/template resolve, resource estimate, quota/admission | 在生成前知道是否支持 snapshot/GPU/network，避免占住模型 worker 后才等待环境 |
| 1. 分配/恢复 | `create`、`claim warm pool`、`resume`、`clone(snapshot)` | 绑定 `rollout_id ↔ sandbox_id ↔ policy version`；预热降低 P99 |
| 2. 初始化 | upload task、write files、env/secret binding、setup `exec`、health check | 任务数据与凭据应最小化；初始化失败不是 policy failure |
| 3. 建立模型 session | 创建 gateway session、返回 scoped model URL/API key、设置 session affinity | 所有 harness 子进程的模型请求必须被训练网关捕获，才能得到准确 token/logprob/mask |
| 4. 多轮 step | `exec/PTY/file/browser/MCP`，同时采集 observation、exit code、tool latency、resource stats | 一个 rollout 内必须共享文件/进程状态；长尾命令不能阻塞整个 batch |
| 5. 中间状态 | `snapshot/pause/heartbeat/set_timeout` | 分叉探索、故障恢复、空闲降本、把环境资源与模型生成解耦 |
| 6. 验证与 reward | 在当前环境验证，或从干净 snapshot/镜像 `clone` 后执行 tests；读取 diff/artifact | 干净 verifier 防止 agent 篡改测试/评测脚本；基础设施错误必须与零 reward 分开 |
| 7. trajectory 结束 | finalize token tree、reward、artifacts；`kill/delete` 或 `pause` 留待复用 | 先持久化轨迹再回收；异常路径也必须回收 orphan |
| 8. 训练消费 | TransferQueue/replay buffer；按 policy version、snapshot lineage 对齐 | 异步训练中防止过旧样本和不同环境版本被混在一起 |

### 6.2 各框架的当前方案

#### Uni-Agent

[Uni-Agent Sandbox 概念文档](https://uni-agent.readthedocs.io/en/latest/concepts/sandbox.html) 明确把 sandbox 定义为 agent episode 的执行边界，由 Task 负责 start/stop。`SandboxBackend` 统一提供 `exec`、`exec_shell`、`read_file`、`write_file`、upload/download、`expose_port`，并用 async context manager 实现重试、启动超时和全局并发限制。provider 包括不隔离的 local、临时 Docker、veFaaS、Modal、OpenYuanrong（[启动说明](https://uni-agent.readthedocs.io/en/latest/quickstart/launch-sandbox.html)）。

它更关键的设计是 Gateway：每个 rollout 创建 scoped session，向 sandbox 内的 Claude Code/Codex 等 harness 注入 OpenAI/Anthropic-compatible model URL；所有模型调用返回到训练侧，从而得到精确 token、mask 和 logprob，Task 再提交 reward，最后通过 TransferQueue 交给 verl（[Gateway 与 trajectory](https://uni-agent.readthedocs.io/en/latest/concepts/gateway-and-trajectories.html)）。这将“机器执行面”和“模型采样面”解耦，是目前较完整的 agent-RL 集成形态。

#### verl

verl 是训练/rollout engine，而不是 sandbox control plane。当前多轮框架通过 `AgentLoop`、`BaseTool`/Interaction 和 per-trajectory state 接入环境；文档要求自定义 tool 在 trajectory 结束时清理 VM、scratch space、DB 等资源（[multi-turn 文档](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)）。其 [AgentLoop tracking issue #2618](https://github.com/verl-project/verl/issues/2618) 反映了 reward 移入 loop、server mode、性能和 tool/interaction 统一等演进；代码 RL 现有集成以 SandboxFusion 为代表（[RFC #5531](https://github.com/verl-project/verl/issues/5531)）。

因此当前实践通常是：verl 管理 policy rollout 与训练，Uni-Agent 或用户自定义 AgentLoop 管理 harness/session，SandboxFusion/Docker/云 provider 执行代码。不要把 `BaseTool.execute()` 当成完整生命周期；大规模部署还需要租约、并发、回收和失败分类。

#### slime

slime 的扩展点是 `custom-generate`、`custom-rm` 与异步 rollout。其 [coding-agent RL 示例](https://thudm.github.io/slime/_examples_synced/coding_agent_rl/README.html) 展示了更具体的流程：每个样本创建新 sandbox，上传任务/仓库，sandbox 内运行 Claude Code 或 Codex，经 E2B-compatible endpoint 调模型；结束后提取 git diff，并在**第二个干净 sandbox**中评分以降低作弊。`slime.agent.sandbox.Sandbox` 提供 `exec/read_file/write_file`，E2BSandbox 是具体实现。

slime 还给同一 rollout 使用稳定 `X-SMG-Routing-Key`，让多轮模型请求保持 prefix-cache affinity；adapter 捕获精确 sampled token id/logprob，并支持消息树分支（[Agent RL 文档](https://thudm.github.io/slime/get_started/agent.html)）。这说明 sandbox placement 与 inference cache routing 需要协同，不能由两个完全无感的调度器各自决定。

#### Miles

[Miles](https://github.com/radixark/miles) 与 slime 同源/持续吸收其训练和 rollout 设计，重点是大规模异步 RL 与低精度训练。截止检索日，在其主分支文档和公开 issue/PR 中未找到像 slime coding-agent example 那样的一等 `Sandbox` provider 抽象或 E2B 示例；现阶段可通过 `custom-generate/custom-rm/data-source` 接入外部环境。这个“未发现”结论只针对当前公开主分支，不能据项目血缘假定 slime 最新 sandbox 模块已存在于 Miles。

#### ROLL + ROCK

ROLL 的 agentic pipeline 把环境级异步 rollout 作为一等场景；其配套 [ROCK](https://github.com/alibaba/ROCK) 是分布式 sandbox/environment 服务。ROCK 默认以 Docker 管理环境，架构含 Admin/Worker/Rocklet；SDK 同时提供 sandbox `start/alive/stats/stop/commit`、命令、文件、持久 bash session，以及 GEM `make/reset/step/close`。这使同一系统既能服务 coding shell，也能服务有显式 action/observation/reward 的交互环境（[ROCK overview](https://alibaba.github.io/ROCK/docs/overview/)、[ROLL 集成指南](https://alibaba.github.io/ROCK/docs/Getting%20Started/rockroll/)）。

其优点是环境控制面独立扩缩容；代价是 Docker 默认 backend 的共享内核边界不适合跨信任域的任意代码，生产多租户需要换 secure runtime 或增加更强节点隔离。

#### AReaL + AEnvironment

AReaL 2.x 的核心是高并发 agent rollout、OpenAI-compatible proxy、精确 token tracking 和异步训练。agent service 可 inline、subprocess 或 online 部署；`start_session` 为每个 rollout 生成 scoped API key，管理 key 不应进入 sandbox（[Agentic RL](https://github.com/areal-project/AReaL/blob/main/docs/en/tutorial/agentic_rl.md)、[CLI reference](https://github.com/areal-project/AReaL/blob/main/docs/en/cli_reference.md)）。2026 年合入的 [PR #1043](https://github.com/areal-project/AReaL/pull/1043) 增加 rollout gateway，[PR #1048](https://github.com/areal-project/AReaL/pull/1048) 增加 Agent Service；[PR #1231](https://github.com/areal-project/AReaL/pull/1231) 又把 Daytona 作为 opt-in cloud sandbox backend 接入：一条路径是每 trajectory 保持解释器状态的 async tool，另一条是 reward/eval/data-prep 用的同步 runner，并显式加入 `aclose` 清理。之后的 [PR #1462](https://github.com/areal-project/AReaL/pull/1462) 给出 SWE-bench RL workflow。这条 PR 演进链比只看 README 更能说明 provider 正从示例层进入正式集成面。

[AEnvironment](https://github.com/inclusionAI/AEnvironment) 则把环境工具、state、reward 统一为扩展 MCP 的 Environment 抽象，当前 engine 以 Kubernetes 为主，覆盖 Terminal/SWE/TAU2 等环境。组合起来的分工是：AReaL 管模型 session/trajectory/训练时序，AEnvironment 或 Daytona/K8s 管环境执行。AReaL 的异步模式让 rollout 与 training overlap（[async 文档](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/async.md)），但也要求 sandbox 状态带 policy/environment version，避免恢复出的旧状态生成无法解释的 off-policy 样本。

### 6.3 框架间的真正差异

| 系统 | Sandbox 抽象位置 | 当前主要 backend/provider | 突出能力 | 主要空缺/风险 |
|---|---|---|---|---|
| Uni-Agent | Task + `SandboxBackend` + Gateway | local/Docker/veFaaS/Modal/OpenYuanrong | harness-agnostic、精确 trajectory、provider 可插拔 | snapshot/fork 不是最小统一面；provider 语义差异 |
| verl | AgentLoop/Tool 的用户扩展 | SandboxFusion/自定义 | 与训练算法、SGLang/vLLM rollout 深度集成 | 不是完整 sandbox control plane，清理和 session 管理靠上层 |
| slime | custom generation + E2B wrapper | E2B-compatible cluster | Claude/Codex coding RL、干净评分 sandbox、prefix affinity | 示例导向，provider/错误语义仍需工程化 |
| Miles | custom generation/reward hooks | 用户自接 | 大规模异步训练主干 | 截止当前未见一等 sandbox SDK 文档 |
| ROLL/ROCK | 独立分布式 env service | 默认 Docker | sandbox API + GEM 环境 API；Admin/Worker | Docker threat boundary；secure backend 的统一能力仍需补齐 |
| AReaL/AEnvironment | model proxy/session + MCP Environment | K8s、Daytona、workflow provider | agent service、async RL、环境/奖励工具化 | 两项目接口边界仍在快速演进；snapshot 语义未形成统一协议 |

---

## 7. AgentENV 与 CubeSandbox：设计、功能与性能

### 7.1 先给结论

两者都以 KVM microVM、OCI image、E2B-compatible API 和 snapshot/resume 为卖点，但优化中心明显不同：

- **AgentENV 是以镜像/块存储/内存共享为中心的 RL sandbox data plane。** Firecracker 负责 VMM，OverlayBD + ublk 负责 OCI layer 按需读取、CoW 与跨实例 page cache，snapshot 可落 S3/共享文件系统，目标是大规模 rollout 的冷启动、磁盘和跨节点分发。
- **CubeSandbox 是端到端 sandbox platform。** 自研 CubeHypervisor 配合 CubeCoW、CubeVS/eBPF、CubeEgress、API/Master/Proxy/Cubelet，强项是集群控制面、网络/secret/审计、SDK/console 以及本地高速 snapshot/clone。

这不是简单的“谁更快”。AgentENV 公布的 `<50 ms boot/resume`、`<100 ms pause/snapshot` 与 Cube 的 `<60 ms create`、约 `100 ms snapshot`、50 并发创建 P95 等数字来自不同机器、镜像、缓存和统计口径，均是项目方数据，只能说明设计目标，不能构成公平排名。

### 7.2 逐项比较

| 维度 | AgentENV | CubeSandbox | 对 RL Infra 的影响 |
|---|---|---|---|
| VMM | Firecracker；复用成熟极简 VMM/jailer/snapshot 机制 | 基于 rust-vmm/KVM 的 CubeHypervisor，与平台数据面协同 | AgentENV VMM 复用风险较低；Cube 有更大端到端优化空间但维护面更广 |
| 镜像 | OCI layer 经 OverlayBD/LSMT 压缩和随机读，ublk 暴露 block device，按需拉取 | OCI/模板 + CubeCoW；本地文件系统优化更突出 | 大模型/大 repo/大量依赖时，AgentENV 的 lazy data path 和共享 page cache 更有价值 |
| 磁盘 snapshot | OverlayBD layer/CoW，可放 S3 或共享 FS | XFS `FICLONE` reflink，节点本地 O(1) clone | AgentENV 更自然支持跨节点对象存储；Cube 本地 clone 极快，但跨节点 resume 仍是公开 issue |
| 内存 snapshot | Firecracker sparse diff memory，恢复时 mmap/CoW；共享 cache | 读取 `/proc/self/pagemap` soft-dirty，只保存脏匿名页，与磁盘一致性 snapshot | 都支持增量状态；需要记录 kernel/VMM/image 兼容版本和 entropy 处理 |
| 网络 | 公开材料更偏存储/运行时；产品级 egress/credential 控制较少 | CubeVS eBPF 网络、CubeEgress L7/TLS 代理、域名策略、credential injection/audit | 运行真实 coding agent 时 Cube 的安全/审计面更完整 |
| 控制面 | 当前项目较新，部署和 API 较精简，E2B compatible | API/Master/Proxy/Cubelet、K8s、配额、Web/SDK、生命周期较完整 | 多团队共享服务时 Cube 更接近成品；科研改造 AgentENV 的数据路径更直接 |
| 安全成熟度 | README 明确**没有 authentication，必须部署在可信网络** | 有租户/网络/secret 设计，但 release 仍持续修复 TAP 回收、跨 sandbox policy 等问题 | 两者都不能只凭“microVM”忽略 control plane；AgentENV 尤其不可直接暴露公网 |
| 硬件/部署 | Linux 6.8+、Ubuntu 24.04、KVM、ublk 等要求较强 | KVM、XFS/reflink 等平台约束；集群组件更多 | 应把 node kernel/fs 能力加入 scheduler label 与 capability discovery |
| 成熟度 | 公开历史较短、提交量少，面向 Kimi K3 RL 场景 | 社区/功能/发行节奏更成熟，2026-07 已到 v0.6.0 | 科研原型与生产平台的风险偏好不同 |

AgentENV 的内部说明见 [README](https://github.com/kvcache-ai/AgentENV) 和 [CLAUDE.md](https://github.com/kvcache-ai/AgentENV/blob/main/CLAUDE.md)。其公开 issue 与后续修复记录暴露了值得研究的真实边角：per-sandbox disk I/O rate limit（[#46](https://github.com/kvcache-ai/AgentENV/issues/46)）、weighted I/O scheduling（[#47](https://github.com/kvcache-ai/AgentENV/issues/47)）、启动失败资源泄漏（[#42](https://github.com/kvcache-ai/AgentENV/issues/42)，已由后续修复关闭）、pause 失败后的状态不一致（[#41](https://github.com/kvcache-ai/AgentENV/issues/41)，已关闭），以及截至检索日仍 open 的 fork 后 entropy/session 未重置（[#33](https://github.com/kvcache-ai/AgentENV/issues/33)）。这些都比平均 boot time 更直接影响 RL 数据质量。

Cube 的 [架构说明](https://cubesandbox.com/architecture/overview)、[snapshot/clone/rollback 深入文章](https://cubesandbox.com/zh/blog/posts/2026-06-25-cubesandbox-snapshot-clone-rollback-deep-dive.html) 和 [生命周期文档](https://cubesandbox.com/guide/lifecycle.html) 说明：pause 会保留 CPU/内存/文件状态，但既有 outbound socket 不保证恢复；资源不足时 resume 可返回冲突；当前跨节点恢复仍在演进（[#1197](https://github.com/TencentCloud/CubeSandbox/issues/1197)）。控制面重启后 paused sandbox 的 TAP 复用问题（[#1207](https://github.com/TencentCloud/CubeSandbox/issues/1207)）也说明 snapshot 正确性必须覆盖网络与控制面状态，而不只是 RAM+disk。

### 7.3 snapshot、pause/resume 对 RL 的直接价值

1. **prefix state 分叉**：在相同仓库、进程、缓存和对话前缀处 snapshot，clone N 个环境采样不同动作，形成严格配对的 counterfactual trajectories。比从头重放更快，也减少环境噪声。
2. **树搜索与 branching policy optimization**：把 snapshot ID 作为搜索树节点，支持 MCTS/best-of-N/backtracking；[Branching Policy Optimization](https://arxiv.org/abs/2607.14171) 已把 sandbox-native branching 用于 agent RL。
3. **干净 verifier**：从 agent 无权修改的 checkpoint clone 出评分环境，只应用 patch/artifact，再运行隐藏测试；slime 的第二 sandbox 是较朴素版本，snapshot 可显著降低成本。
4. **课程学习与难例复用**：保存失败前的中间状态，构造“从这里继续”的短 horizon task；可以研究状态级 curriculum，而不是只按完整题目采样。
5. **容错和抢占**：长尾 rollout pause 后释放部分 CPU/内存，在模型权重更新、队列拥塞或节点维护后 resume；避免整条长 trajectory 作废。
6. **模型与环境解耦调度**：LLM 正在长时间思考时暂停环境；模型完成后恢复。进一步可在模型 decode 期间投机预热 sandbox，[SpecBox](https://arxiv.org/abs/2607.23933) 探索的正是 generation 与 sandbox setup overlap。
7. **精确重放和系统调试**：checkpoint + action log + image/kernel/policy version 可复现一次 reward 异常；对 flaky tests、环境污染和基础设施错误做因果定位。
8. **数据去重和增量存储**：大量 rollout 共享 base image、repo 和前缀内存页；CoW/dedup 可显著降低存储与网络。Delta checkpoint/rollback 的研究如 [DeltaBox](https://arxiv.org/abs/2605.22781) 报告了面向 agent workflow 的毫秒级 checkpoint/rollback，但仍需在真实 RL workload 上复现。

### 7.4 值得 infra 科研工作者探索的系统问题

**A. snapshot-aware rollout scheduler。** 联合优化 GPU decode、sandbox create/resume、snapshot locality、node page cache 和 P99；目标函数不再只是 GPU utilization，而是“每美元/每秒产出的有效、不过期 trajectory token”。

**B. branching-aware storage。** 用 DAG 表达 `base image → task init → turn checkpoint → N branches`，按引用计数和 reward 价值回收层；研究 memory/disk delta 的跨节点放置、压缩、prefetch 与 cache admission。

**C. semantic checkpoint。** OS-level snapshot 捕获字节状态，却不知道“一个 agent turn 是否完成、外部 API 是否可重放”。[Crab](https://arxiv.org/abs/2604.28138) 指出了 OS checkpoint 与 agent 语义之间的差距。可研究 tool transaction boundary、外部副作用日志、幂等补偿和一致性 cut。

**D. paired/counterfactual RL。** 同一 checkpoint fork 不同 policy version、temperature 或 tool policy，做低方差 A/B、advantage estimate、causal credit assignment；同时控制 fork 后 RNG/entropy 独立性。

**E. clean-room reward service。** verifier 从不可写 checkpoint 创建，secret 仅在 egress proxy 注入，测试结果带 attestation；研究如何防测试泄漏、reward tampering 与 artifact substitution。

**F. pause-aware asynchronous RL。** 根据 policy staleness、预计剩余 horizon、sandbox state size 和 GPU queue 决定继续、pause、迁移还是丢弃；把“过期 trajectory 的统计价值”纳入资源调度。

**G. 分层隔离的自适应选择。** 不是所有 rollout 都需 microVM：静态分析/纯 Python 可用 Wasm，可信 compiler task 用 gVisor，陌生 repo/privileged build 用 microVM。可按任务风险和 syscall profile 动态选 backend，并用统一接口保持训练代码不变。

### 7.5 必须先解决的正确性与安全陷阱

- **fork 后随机性相关**：复制 `/dev/urandom`、TLS/session state 或 language runtime PRNG 会让不同分支相关，甚至重复密钥；AgentENV issue #33 是真实例子。
- **外部世界不能随 RAM 回滚**：网络请求、数据库写入、GitHub comment、包仓库 mutable tag 不会自动撤销；需要网络 mock、幂等 key 或副作用日志。
- **snapshot 可能保存 secret**：环境变量、shell history、代理 token、内存中的凭据会进入 snapshot；必须加密、租户绑定、TTL、revoke 和安全删除。
- **连接恢复语义**：TCP、PTY、FUSE、GPU context、挂载租约可能在 resume 后失效；API 应返回 capability/partial-restore 状态，而不是假装透明。
- **版本兼容**：VMM、kernel、CPU feature、guest agent、image layer、network policy 任一变化都可能令 snapshot 不可恢复；必须内容寻址并记录 compatibility tuple。
- **resume storm 与资源超卖**：pause 释放多少资源必须明确定义；批量 resume 可能同时申请 RAM/IO/网络，应有 admission control 和 backpressure。
- **训练统计污染**：sandbox crash/infra timeout 不应当作 agent 得到零 reward；错误 taxonomy 和 retry policy 是算法正确性的一部分。

---

## 8. 选择建议

| 需求 | 首选起点 | 理由 |
|---|---|---|
| 本地单用户 coding agent，低延迟、直接改当前 repo | bubblewrap/Seatbelt + deny-by-default egress + 精确 mount | 原生 UX 和最低开销；用户仍应 review diff |
| 内部 CI、语言评测、可信研究集群 | rootless Docker/runc，必要时 gVisor | OCI 生态和密度好；不要挂 host Docker socket |
| K8s 上多团队不可信代码 | Agent Sandbox CRD + gVisor 或 Kata | 声明式 lifecycle/warm pool 与 stronger runtime 解耦 |
| 公有云多租户、任意 coding/browser agent | Firecracker/Cloud Hypervisor 类 microVM + host egress/secret proxy | 独立 kernel、完整 Linux、snapshot；控制面仍需 hardening |
| 完整 Windows/桌面/企业软件 | 完整 VM | 兼容性优先，接受成本 |
| 受控函数/插件/纯数据工具 | Wasm/WASI | capability 最小、启动和密度好 |
| 大规模 agent RL，镜像/状态重复度高 | microVM + prewarm + incremental snapshot/clone；AgentENV/Cube/E2B 类 API | 既要不可信代码隔离，又要 session state、分叉和故障恢复 |

如果目标是为一个新的 RL Infra 设计统一层，建议不要把抽象绑定到 `DockerContainer` 或 `FirecrackerVM`。最小对象应是带 lease 的 `SandboxSession`，明确 capability、policy、version、lineage 和 failure taxonomy；后端只负责实现其支持的子集。第一版优先做好 create/exec/file/kill、限额、幂等和错误分类，再加入 snapshot/clone。错误的 snapshot 语义比没有 snapshot 更危险。

---

## 9. 资料与证据质量说明

1. Firecracker NSDI 论文、gVisor 架构/性能文档、Wasm/WASI 安全模型属于原理和边界的一手资料；项目 release、代码与 issue 用于判断 2026 年当前实现状态。
2. AgentENV、CubeSandbox、E2B、Vercel 等项目给出的启动/内存数字是各自环境中的项目方结果。本文没有把它们画成统一性能排行榜。
3. 2026 年关于 Agent Sandbox 的论文中相当一部分仍是 arXiv preprint，例如 [SandboxEscapeBench](https://arxiv.org/abs/2603.02277)、[Agent Sandbox isolation survey](https://arxiv.org/abs/2607.12406)、[rollout infrastructure tax](https://arxiv.org/abs/2607.01415)；它们适合发现研究问题，不应等同于经过多年生产验证的安全保证。
4. GitHub issue 表示维护者和用户观察到的问题或规划，不必然表示所有版本均受影响；本文使用它们展示设计边角和当前演进方向。
5. “未发现”只表示在截止日期的公开主分支、文档和可检索 issue/PR 中未找到，不表示私有部署或未来版本不存在。
