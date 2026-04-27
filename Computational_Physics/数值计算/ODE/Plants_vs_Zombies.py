import random
import tkinter as tk
from dataclasses import dataclass
import json
import os
import threading
import urllib.error
import urllib.request

"""
简化版植物大战僵尸（Tkinter 实现）

阅读建议：
1. 先看常量区，理解棋盘尺寸和三种植物参数。
2. 再看三个 dataclass，它们是游戏中的核心实体（植物、僵尸、子弹）。
3. 最后按 PVZGame 的调用链看：
   loop -> update_game -> (update_plants/update_peas/update_zombies) -> draw
"""


# ===== 棋盘与窗口配置 =====
ROWS = 5
COLS = 9
CELL_W = 90
CELL_H = 95
TOP_BAR = 80
SIDE_BAR = 220
WIDTH = COLS * CELL_W + SIDE_BAR
HEIGHT = ROWS * CELL_H + TOP_BAR

# ===== 植物类型标识 =====
PEASHOOTER = "peashooter"
SUNFLOWER = "sunflower"
WALLNUT = "wallnut"

# ===== 植物阳光消耗 =====
PLANT_COST = {
    PEASHOOTER: 100,
    SUNFLOWER: 50,
    WALLNUT: 75,
}


@dataclass
class Plant:
    """植物实体。

    `timer` 用于该植物自身的周期行为：
    - 豌豆射手：定时发射豌豆
    - 向日葵：定时产出阳光
    """

    kind: str
    row: int
    col: int
    hp: int
    timer: int = 0


@dataclass
class Zombie:
    """僵尸实体。

    `x` 是连续坐标（浮点数），每帧向左移动，实现平滑动画。
    `bite_timer` 控制啃咬间隔，避免每帧都造成伤害。
    """

    row: int
    x: float
    hp: int
    speed: float
    bite_timer: int = 0


@dataclass
class Pea:
    """豌豆子弹实体。

    子弹只沿水平方向飞行，所以只需要 `row + x + y` 即可定位。
    """

    row: int
    x: float
    y: float
    speed: float = 9.0
    damage: int = 25


class PVZGame:
    @staticmethod
    def load_local_env_file():
        """读取脚本同目录下的 .env（若存在），返回键值映射。"""
        env_map = {}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, ".env")
        if not os.path.exists(env_path):
            return env_map

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    env_map[key.strip()] = value.strip()
        except Exception:
            # 配置读取失败时回退到系统环境变量，不阻断游戏
            return {}

        return env_map

    def __init__(self, root: tk.Tk):
        """初始化游戏状态、输入绑定，并启动主循环。"""
        self.root = root
        self.root.title("Plants vs Zombies - Simplified")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#7CCB5E", highlightthickness=0)
        self.canvas.pack()

        # 运行时实体容器：
        # plants: {(row, col): Plant}
        # zombies: Zombie 列表
        # peas: Pea 列表
        self.plants = {}
        self.zombies = []
        self.peas = []

        # 资源与游戏状态
        self.sun = 150
        self.selected = PEASHOOTER
        self.running = True
        self.elapsed_ms = 0

        # 各种全局计时器
        self.spawn_timer = 0
        self.passive_sun_timer = 0

        # 存活目标时长：120 秒
        self.target_survival_ms = 120000

        # 自动战斗开关：默认开启，可按 T 键随时切换
        self.auto_mode = True
        self.auto_timer = 0
        self.ai_request_timer = 0
        local_env = self.load_local_env_file()

        # 自动模式后端：
        # - heuristic: 本地启发式策略（默认）
        # - codex: 调用 OpenAI Responses API 让模型给出操作
        # 优先级：系统环境变量 > 同目录 .env > 默认值
        self.auto_backend = os.getenv("PVZ_AGENT", local_env.get("PVZ_AGENT", "heuristic")).strip().lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", local_env.get("OPENAI_API_KEY", "")).strip()
        self.openai_model = os.getenv("OPENAI_MODEL", local_env.get("OPENAI_MODEL", "gpt-5")).strip()

        # 仅当用户显式选择 codex 且提供 API Key 时才启用模型代打
        self.use_codex_agent = self.auto_backend == "codex" and bool(self.openai_api_key)
        self.ai_request_in_flight = False
        self.ai_suggested_action = None
        self.ai_last_error = ""
        backend_name = "CODEX" if self.use_codex_agent else "HEURISTIC"
        self.root.title(f"Plants vs Zombies - Simplified ({backend_name})")

        # 输入绑定：鼠标种植 + 键盘选卡/重开
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("1", lambda _e: self.select(PEASHOOTER))
        self.root.bind("2", lambda _e: self.select(SUNFLOWER))
        self.root.bind("3", lambda _e: self.select(WALLNUT))
        self.root.bind("r", lambda _e: self.reset())
        self.root.bind("t", lambda _e: self.toggle_auto_mode())
        self.root.bind("T", lambda _e: self.toggle_auto_mode())

        self.loop()

    def reset(self):
        """重开游戏：清空战场并恢复初始资源、计时器和状态。"""
        self.plants.clear()
        self.zombies.clear()
        self.peas.clear()
        self.sun = 150
        self.selected = PEASHOOTER
        self.running = True
        self.elapsed_ms = 0
        self.spawn_timer = 0
        self.passive_sun_timer = 0
        self.auto_timer = 0
        self.ai_request_timer = 0
        self.ai_suggested_action = None
        self.ai_request_in_flight = False
        self.ai_last_error = ""

    def select(self, kind: str):
        """切换当前准备种植的植物类型。"""
        self.selected = kind

    def toggle_auto_mode(self):
        """切换自动战斗状态。"""
        self.auto_mode = not self.auto_mode
        self.auto_timer = 0

    def grid_to_xy(self, row: int, col: int):
        """把网格坐标(row, col)转换成像素矩形(x1, y1, x2, y2)。"""
        x1 = col * CELL_W + 4
        y1 = TOP_BAR + row * CELL_H + 4
        x2 = (col + 1) * CELL_W - 4
        y2 = TOP_BAR + (row + 1) * CELL_H - 4
        return x1, y1, x2, y2

    def try_place_plant(self, row: int, col: int, kind: str):
        """尝试在指定格子种指定植物；成功返回 True，失败返回 False。"""
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return False
        if (row, col) in self.plants:
            return False

        cost = PLANT_COST[kind]
        if self.sun < cost:
            return False

        self.sun -= cost
        if kind == PEASHOOTER:
            plant = Plant(kind=PEASHOOTER, row=row, col=col, hp=100, timer=0)
        elif kind == SUNFLOWER:
            plant = Plant(kind=SUNFLOWER, row=row, col=col, hp=80, timer=0)
        else:
            plant = Plant(kind=WALLNUT, row=row, col=col, hp=320, timer=0)
        self.plants[(row, col)] = plant
        return True

    def on_click(self, event):
        """处理鼠标点击：在草坪格子内尝试种植当前选中的植物。"""
        if not self.running:
            # 结束状态下禁用种植
            return

        if event.x >= COLS * CELL_W or event.y < TOP_BAR:
            # 点到侧边栏或顶部 UI 区，忽略
            return

        col = event.x // CELL_W
        row = (event.y - TOP_BAR) // CELL_H
        if not (0 <= row < ROWS and 0 <= col < COLS):
            # 安全边界校验
            return

        key = (row, col)
        if key in self.plants:
            # 该格已有植物
            return

        self.try_place_plant(row=row, col=col, kind=self.selected)

    def spawn_zombie(self):
        """随机生成一只僵尸并放在最右侧场外，随后向左进入战场。"""
        row = random.randint(0, ROWS - 1)
        hp = random.randint(120, 180)
        speed = random.uniform(0.25, 0.45)
        self.zombies.append(Zombie(row=row, x=COLS * CELL_W + 40, hp=hp, speed=speed))

    def update_plants(self, dt: int):
        """更新植物的周期行为（开火/产阳光）。"""
        for plant in list(self.plants.values()):
            plant.timer += dt
            if plant.kind == PEASHOOTER:
                if plant.timer >= 1200:
                    # 豌豆射手每 1.2 秒发射一颗子弹
                    y = TOP_BAR + plant.row * CELL_H + CELL_H / 2
                    x = plant.col * CELL_W + CELL_W * 0.72
                    self.peas.append(Pea(row=plant.row, x=x, y=y))
                    plant.timer = 0
            elif plant.kind == SUNFLOWER:
                if plant.timer >= 5200:
                    # 向日葵每 5.2 秒提供 25 阳光
                    self.sun += 25
                    plant.timer = 0

    def update_peas(self):
        """更新豌豆移动与碰撞，命中后扣僵尸血并销毁子弹。"""
        for pea in list(self.peas):
            pea.x += pea.speed

            if pea.x > COLS * CELL_W + 40:
                # 飞出右边界后移除
                self.peas.remove(pea)
                continue

            hit = False
            for zombie in self.zombies:
                if zombie.row != pea.row:
                    # 只判定同一行碰撞
                    continue
                if abs(zombie.x - pea.x) <= 18:
                    # 用简单的一维距离阈值判定命中
                    zombie.hp -= pea.damage
                    hit = True
                    break
            if hit and pea in self.peas:
                self.peas.remove(pea)

        # 过滤掉血量<=0 的僵尸
        self.zombies = [z for z in self.zombies if z.hp > 0]

    def zombie_target_plant(self, zombie: Zombie):
        """返回僵尸当前可啃咬的目标植物；若没有则返回 None。

        说明：
        - 僵尸位置是连续坐标，不总是精确落在某一列中心。
        - 这里会同时检查当前列和左一列，减少边界抖动导致的“穿模”。
        """
        col = int(zombie.x // CELL_W)
        candidates = []
        for offset in (0, -1):
            c = col + offset
            if 0 <= c < COLS:
                plant = self.plants.get((zombie.row, c))
                if plant is not None:
                    px = c * CELL_W + CELL_W / 2
                    if zombie.x <= px + CELL_W * 0.45:
                        candidates.append((c, plant))
        if not candidates:
            return None
        # 选择列号更靠右的目标，优先啃咬最近撞到的植物
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def update_zombies(self, dt: int):
        """更新僵尸移动/啃咬逻辑，并判定失败条件（僵尸进家）。"""
        for zombie in list(self.zombies):
            target = self.zombie_target_plant(zombie)
            if target is None:
                # 前方无植物，继续向左移动
                zombie.x -= zombie.speed * dt
            else:
                # 有目标时停止前进，进入啃咬计时
                zombie.bite_timer += dt
                if zombie.bite_timer >= 700:
                    # 每 700ms 对目标造成一次伤害
                    target.hp -= 20
                    zombie.bite_timer = 0
                    if target.hp <= 0:
                        # 植物死亡后从字典中移除
                        self.plants.pop((target.row, target.col), None)

            if zombie.x < 0:
                # 任意僵尸越过左边界即判负
                self.running = False

    def build_compact_state(self):
        """构造发送给模型的紧凑状态，降低 token 开销与延迟。"""
        plants = []
        for plant in self.plants.values():
            plants.append(
                {
                    "kind": plant.kind,
                    "row": plant.row,
                    "col": plant.col,
                    "hp": plant.hp,
                }
            )

        zombies = []
        for zombie in self.zombies:
            zombies.append(
                {
                    "row": zombie.row,
                    "x": round(zombie.x, 1),
                    "hp": zombie.hp,
                    "speed": round(zombie.speed, 3),
                }
            )

        return {
            "rows": ROWS,
            "cols": COLS,
            "sun": self.sun,
            "elapsed_ms": self.elapsed_ms,
            "target_survival_ms": self.target_survival_ms,
            "plants": plants,
            "zombies": zombies,
            "cost": PLANT_COST,
            "allowed_kinds": [PEASHOOTER, SUNFLOWER, WALLNUT],
        }

    def extract_json_object(self, text: str):
        """从模型返回文本中提取第一个 JSON 对象。"""
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def normalize_ai_action(self, action):
        """校验并标准化模型动作，非法动作返回 {'action': 'none'}。"""
        if not isinstance(action, dict):
            return {"action": "none"}

        raw_action = str(action.get("action", "none")).strip().lower()
        if raw_action != "plant":
            return {"action": "none"}

        kind = str(action.get("kind", "")).strip().lower()
        row = action.get("row")
        col = action.get("col")
        if kind not in (PEASHOOTER, SUNFLOWER, WALLNUT):
            return {"action": "none"}
        if not isinstance(row, int) or not isinstance(col, int):
            return {"action": "none"}
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return {"action": "none"}

        return {"action": "plant", "kind": kind, "row": row, "col": col}

    def fetch_codex_action_worker(self, state):
        """后台线程：调用 OpenAI API 获取下一步操作。"""
        try:
            developer_prompt = (
                "你是植物大战僵尸代打代理。"
                "只允许返回一个 JSON 对象，不要返回 markdown。"
                "JSON 格式固定为："
                '{"action":"plant","kind":"peashooter|sunflower|wallnut","row":0,"col":0}'
                "或"
                '{"action":"none"}。'
                "优先保证防线，再补经济和输出。"
            )

            # 尝试模型列表：优先使用用户配置模型，失败后回退到兼容模型。
            # 这样在账户没有某个模型权限时，仍能继续代打。
            model_candidates = [self.openai_model, "gpt-4.1-mini", "gpt-4.1"]
            # 去重并保序
            dedup_models = []
            for m in model_candidates:
                if m and m not in dedup_models:
                    dedup_models.append(m)

            data = None
            last_http_error = None
            last_error_detail = ""
            for model_name in dedup_models:
                payload = {
                    "model": model_name,
                    "input": [
                        {
                            "role": "developer",
                            "content": [{"type": "input_text", "text": developer_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "当前游戏状态(JSON): " + json.dumps(state, ensure_ascii=False),
                                }
                            ],
                        },
                    ],
                    "max_output_tokens": 120,
                }

                req = urllib.request.Request(
                    url="https://api.openai.com/v1/responses",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.openai_api_key}",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        body = resp.read().decode("utf-8")
                    data = json.loads(body)
                    # 成功时记录实际使用模型，方便 UI 对齐真实情况
                    self.openai_model = model_name
                    break
                except urllib.error.HTTPError as err:
                    last_http_error = err
                    detail = ""
                    try:
                        err_body = err.read().decode("utf-8")
                        err_json = json.loads(err_body)
                        # OpenAI 常见错误结构：{"error": {"message": "..."}}
                        detail = str(err_json.get("error", {}).get("message", "")).strip()
                    except Exception:
                        detail = ""
                    last_error_detail = detail or f"HTTP {err.code}"
                    # 4xx 通常是权限/模型/配额问题，继续尝试下一个回退模型
                    continue

            if data is None:
                if last_http_error is not None:
                    code = last_http_error.code
                    self.ai_last_error = f"HTTP {code}: {last_error_detail}"
                else:
                    self.ai_last_error = "All model attempts failed"
                self.ai_suggested_action = {"action": "none"}
                return

            # 兼容两种返回：优先 output_text；否则从 output[].content[] 取文本
            text = data.get("output_text", "")
            if not text:
                text_parts = []
                for item in data.get("output", []):
                    for content in item.get("content", []):
                        t = content.get("text")
                        if isinstance(t, str):
                            text_parts.append(t)
                text = "\n".join(text_parts)

            parsed = self.extract_json_object(text)
            self.ai_suggested_action = self.normalize_ai_action(parsed)
            self.ai_last_error = ""
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                err_body = err.read().decode("utf-8")
                err_json = json.loads(err_body)
                detail = str(err_json.get("error", {}).get("message", "")).strip()
            except Exception:
                detail = ""
            self.ai_last_error = f"HTTP {err.code}: {detail or 'request failed'}"
            self.ai_suggested_action = {"action": "none"}
        except Exception as err:
            self.ai_last_error = str(err)
            self.ai_suggested_action = {"action": "none"}
        finally:
            self.ai_request_in_flight = False

    def request_codex_action_async(self):
        """异步发起模型决策请求，避免阻塞 Tk 主线程。"""
        if self.ai_request_in_flight:
            return
        self.ai_request_in_flight = True
        state = self.build_compact_state()
        thread = threading.Thread(target=self.fetch_codex_action_worker, args=(state,), daemon=True)
        thread.start()

    def apply_ai_action_if_ready(self):
        """如果模型动作已返回，则尝试执行一次。"""
        action = self.ai_suggested_action
        if not isinstance(action, dict):
            return
        self.ai_suggested_action = None

        if action.get("action") != "plant":
            return
        self.try_place_plant(row=action["row"], col=action["col"], kind=action["kind"])

    def auto_battle_step(self):
        """自动战斗单步决策。

        策略是启发式而非最优解，优先级如下：
        1) 紧急防线：危险行补坚果墙
        2) 火力覆盖：有僵尸的行先保证至少一个豌豆射手
        3) 经济建设：逐步补向日葵
        4) 中后期补强：继续叠加豌豆射手
        """
        if not self.running or not self.auto_mode:
            return
        if self.use_codex_agent:
            # 先应用已返回的动作，再按节流间隔请求下一步
            self.apply_ai_action_if_ready()
            if not self.ai_request_in_flight and self.ai_request_timer >= 1800:
                self.request_codex_action_async()
                self.ai_request_timer = 0
            return

        # 将僵尸按行分组，后续按行做针对性决策
        zombies_by_row = {r: [] for r in range(ROWS)}
        for zombie in self.zombies:
            zombies_by_row[zombie.row].append(zombie)

        for r in range(ROWS):
            zombies_by_row[r].sort(key=lambda z: z.x)

        # 1) 紧急防线：如果僵尸已经接近中左区域，优先在该行放置坚果墙
        for row in range(ROWS):
            if not zombies_by_row[row]:
                continue
            nearest = zombies_by_row[row][0]
            if nearest.x < 6.4 * CELL_W:
                for col in (6, 5, 4, 3, 2):
                    plant = self.plants.get((row, col))
                    if plant is None and self.try_place_plant(row=row, col=col, kind=WALLNUT):
                        return
                    if plant is not None and plant.kind == WALLNUT:
                        break

        # 2) 火力覆盖：有僵尸的行，至少要有一个豌豆射手
        for row in range(ROWS):
            if not zombies_by_row[row]:
                continue
            has_pea = any(
                plant.row == row and plant.kind == PEASHOOTER
                for plant in self.plants.values()
            )
            if has_pea:
                continue
            for col in (2, 1, 3, 4):
                if self.try_place_plant(row=row, col=col, kind=PEASHOOTER):
                    return

        # 3) 经济建设：优先在左侧两列种向日葵，逐步把每行补齐
        sunflower_count = sum(1 for plant in self.plants.values() if plant.kind == SUNFLOWER)
        if sunflower_count < 8:
            for row in range(ROWS):
                for col in (0, 1):
                    if self.try_place_plant(row=row, col=col, kind=SUNFLOWER):
                        return

        # 4) 中后期补强：在每行追加豌豆射手，形成持续输出
        for row in range(ROWS):
            for col in (2, 3, 4, 5):
                if self.try_place_plant(row=row, col=col, kind=PEASHOOTER):
                    return

    def update_game(self, dt: int):
        """推进一帧游戏状态（核心逻辑调度函数）。"""
        if not self.running:
            return

        self.elapsed_ms += dt

        # 1) 刷怪：随时间推进逐渐加快（下限 900ms）
        self.spawn_timer += dt
        spawn_interval = max(900, 2400 - self.elapsed_ms // 120)
        if self.spawn_timer >= spawn_interval:
            self.spawn_zombie()
            self.spawn_timer = 0

        # 2) 被动阳光：每 7.8 秒 +25
        self.passive_sun_timer += dt
        if self.passive_sun_timer >= 7800:
            self.sun += 25
            self.passive_sun_timer = 0

        # 自动战斗每 280ms 执行一次决策，避免每帧都“抢种”导致行为突兀
        self.auto_timer += dt
        self.ai_request_timer += dt
        if self.auto_mode and self.auto_timer >= 280:
            self.auto_battle_step()
            self.auto_timer = 0

        # 3) 更新三类实体
        self.update_plants(dt)
        self.update_peas()
        self.update_zombies(dt)

        # 4) 达到目标时长则判定通关（running=False，进入结束画面）
        if self.elapsed_ms >= self.target_survival_ms:
            self.running = False

    def draw_ui(self):
        """绘制顶部信息栏与右侧选卡栏。"""
        self.canvas.create_rectangle(0, 0, WIDTH, TOP_BAR, fill="#264653", outline="")
        self.canvas.create_text(18, 24, text=f"Sun: {self.sun}", fill="#FFD166", anchor="w", font=("Arial", 18, "bold"))
        sec = self.elapsed_ms // 1000
        target_sec = self.target_survival_ms // 1000
        self.canvas.create_text(18, 56, text=f"Time: {sec}s / {target_sec}s", fill="white", anchor="w", font=("Arial", 12, "bold"))

        card_x = COLS * CELL_W + 20
        cards = [
            (PEASHOOTER, "1 - Peashooter", PLANT_COST[PEASHOOTER], "#8EEC5C"),
            (SUNFLOWER, "2 - Sunflower", PLANT_COST[SUNFLOWER], "#FFE66D"),
            (WALLNUT, "3 - Wallnut", PLANT_COST[WALLNUT], "#C89B7B"),
        ]
        for i, (kind, label, cost, color) in enumerate(cards):
            y1 = TOP_BAR + 20 + i * 95
            y2 = y1 + 80
            border = "#111111" if self.selected == kind else "#555555"
            self.canvas.create_rectangle(card_x, y1, card_x + 180, y2, fill=color, outline=border, width=4 if self.selected == kind else 2)
            self.canvas.create_text(card_x + 10, y1 + 22, text=label, anchor="w", fill="#1d1d1d", font=("Arial", 12, "bold"))
        self.canvas.create_text(card_x + 10, y1 + 52, text=f"Cost: {cost}", anchor="w", fill="#1d1d1d", font=("Arial", 11))

        self.canvas.create_text(card_x, HEIGHT - 80, text="Click lawn to plant", anchor="w", fill="#f2f2f2", font=("Arial", 11))
        self.canvas.create_text(card_x, HEIGHT - 60, text="Press 1/2/3 to switch", anchor="w", fill="#f2f2f2", font=("Arial", 11))
        self.canvas.create_text(card_x, HEIGHT - 40, text="Press T to toggle auto", anchor="w", fill="#f2f2f2", font=("Arial", 11))
        self.canvas.create_text(card_x, HEIGHT - 20, text="Press R to restart", anchor="w", fill="#f2f2f2", font=("Arial", 11))

        if self.use_codex_agent:
            backend = "CODEX"
        else:
            backend = "HEURISTIC"

        auto_text = f"AUTO: {'ON' if self.auto_mode else 'OFF'} ({backend})"
        auto_color = "#80ED99" if self.auto_mode else "#F28482"
        self.canvas.create_text(card_x, TOP_BAR + 330, text=auto_text, anchor="w", fill=auto_color, font=("Arial", 12, "bold"))
        if self.use_codex_agent and self.ai_last_error:
            self.canvas.create_text(card_x, TOP_BAR + 352, text=f"AI Error: {self.ai_last_error}", anchor="w", fill="#FFD6A5", font=("Arial", 10))

    def draw_grid(self):
        """绘制草坪网格背景。"""
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1, x2, y2 = self.grid_to_xy(r, c)
                color = "#7EC850" if (r + c) % 2 == 0 else "#73BE49"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#5aa23c")

    def draw_plants(self):
        """绘制植物（用简单几何图形做示意）。"""
        for plant in self.plants.values():
            x1, y1, x2, y2 = self.grid_to_xy(plant.row, plant.col)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if plant.kind == PEASHOOTER:
                self.canvas.create_oval(x1 + 10, y1 + 14, x2 - 10, y2 - 8, fill="#2A9D8F", outline="#1f6f67", width=2)
                self.canvas.create_oval(cx + 8, cy - 10, cx + 30, cy + 12, fill="#2A9D8F", outline="#1f6f67", width=2)
                self.canvas.create_oval(cx + 22, cy - 2, cx + 28, cy + 4, fill="#103d38", outline="")
            elif plant.kind == SUNFLOWER:
                self.canvas.create_oval(x1 + 10, y1 + 12, x2 - 10, y2 - 10, fill="#F4A261", outline="#bb783f", width=2)
                self.canvas.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, fill="#FFD166", outline="#c99d35", width=2)
            else:
                self.canvas.create_oval(x1 + 8, y1 + 10, x2 - 8, y2 - 8, fill="#B08968", outline="#7f5f47", width=2)

    def draw_zombies(self):
        """绘制僵尸与血条。"""
        for zombie in self.zombies:
            y1 = TOP_BAR + zombie.row * CELL_H + 10
            y2 = y1 + CELL_H - 20
            x1 = zombie.x - 22
            x2 = zombie.x + 22
            self.canvas.create_rectangle(x1, y1 + 8, x2, y2, fill="#7D8C8D", outline="#4c5757", width=2)
            self.canvas.create_oval(x1 + 6, y1 - 8, x2 - 6, y1 + 18, fill="#A3B18A", outline="#6d785e", width=2)

            hp_ratio = max(0, zombie.hp) / 180
            self.canvas.create_rectangle(x1, y1 - 16, x2, y1 - 10, fill="#2d2d2d", outline="")
            self.canvas.create_rectangle(x1, y1 - 16, x1 + (x2 - x1) * hp_ratio, y1 - 10, fill="#E63946", outline="")

    def draw_peas(self):
        """绘制飞行中的豌豆子弹。"""
        for pea in self.peas:
            self.canvas.create_oval(pea.x - 6, pea.y - 6, pea.x + 6, pea.y + 6, fill="#A7D129", outline="#6b8a15")

    def draw_end_state(self):
        """游戏结束时绘制半透明遮罩和结果文本。"""
        if self.running:
            return
        overlay = "#111111"
        self.canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill=overlay, stipple="gray50", outline="")
        win = self.elapsed_ms >= self.target_survival_ms
        text = "You Win!" if win else "Game Over"
        sub = "Press R to restart"
        self.canvas.create_text(WIDTH / 2, HEIGHT / 2 - 20, text=text, fill="white", font=("Arial", 42, "bold"))
        self.canvas.create_text(WIDTH / 2, HEIGHT / 2 + 28, text=sub, fill="#f1f1f1", font=("Arial", 16))

    def draw(self):
        """整帧重绘入口：按背景->实体->UI->结束态的顺序绘制。"""
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_plants()
        self.draw_peas()
        self.draw_zombies()
        self.draw_ui()
        self.draw_end_state()

    def loop(self):
        """固定时间步长主循环（40ms 一帧，约 25 FPS）。"""
        dt = 40
        self.update_game(dt)
        self.draw()
        self.root.after(dt, self.loop)


def main():
    """程序入口：创建窗口并启动 Tk 事件循环。"""
    root = tk.Tk()
    PVZGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
