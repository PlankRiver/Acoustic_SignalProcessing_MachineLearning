import random
import time

def play_jump_game():
    """
    A text-based version of the "Jump Jump" game.
    """
    print("--- 欢迎来到文本版'跳一跳'游戏! ---")
    print("游戏规则：输入一个数字作为你的'蓄力值'，它将决定你跳跃的距离。")
    print("目标是精确地跳到下一个方块上。祝你好运！\n")
    time.sleep(2)

    # Game setup
    score = 0
    current_position_index = 0
    # Let's generate the first 10 block positions
    blocks = [0]
    position_so_far = 0
    for _ in range(100): # Generate a long path
        distance = random.randint(5, 25)
        position_so_far += distance
        blocks.append(position_so_far)

    # --- Helper function to visualize the board ---
    def draw_board(player_idx, all_blocks):
        # Determine the window to display
        start_pos = all_blocks[player_idx]
        if player_idx + 1 < len(all_blocks):
            end_pos = all_blocks[player_idx+1]
        else:
            end_pos = start_pos + 20 # End of the game

        # Create the board string
        board_list = ['-'] * (end_pos - start_pos + 5)

        # Place the player
        board_list[0] = 'P'

        # Place the next block
        next_block_relative_pos = all_blocks[player_idx+1] - start_pos
        if next_block_relative_pos < len(board_list):
            board_list[next_block_relative_pos] = '■'

        print("".join(board_list))


    # --- Main game loop ---
    while current_position_index < len(blocks) - 1:
        current_pos = blocks[current_position_index]
        next_pos = blocks[current_position_index + 1]
        distance_to_next = next_pos - current_pos

        print(f"\n--- 分数: {score} ---")
        print(f"你现在在位置 {current_pos}。")
        draw_board(current_position_index, blocks)
        print(f"下一个方块在 {distance_to_next} 距离之外。")

        try:
            # The relationship between power and distance. Let's make it simple: 1 to 1.
            power_input = input("请输入你的蓄力值 (1-30): ")
            jump_distance = int(power_input)

            print(f"你蓄力 {jump_distance}, 跳跃...")
            time.sleep(1)

            if jump_distance == distance_to_next:
                print(f"漂亮! 成功跳到位置 {next_pos}!")
                score += 1
                current_position_index += 1
            else:
                print(f"哎呀! 你跳了 {jump_distance} 远, 但目标距离是 {distance_to_next}。")
                print(f"你掉下去了! 游戏结束。")
                break
        except ValueError:
            print("无效输入! 请输入一个数字。游戏结束。")
            break
        except (KeyboardInterrupt, EOFError):
            print("\n游戏已退出。")
            break


    print("\n--- 游戏结束 ---")
    print(f"你的最终分数是: {score}")

# Execute the game
play_jump_game()