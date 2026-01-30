#condition1: use a greedy algorithm
#condition2: only consider smaller than 1000 yen at the amount
#condition3: 

# def coin_num(amount):
#     amount%=1000
#     coin_count=[]
#     for coin in [500,100,50,10,5,1]:
#         coin_count.append(amount//coin)
#     #        if coin_count>budget[coin]:
            
#         amount%=coin
#         print(f"{coin}: {coin_count}")
#     print(coin_count)
# if __name__ == "__main__":
#         budget0={500:1,100:5,50:1,10:5,5:5,1:10}
#         budget1={500:3,100:2,50:0,10:5,5:5,1:10}
#         coin_num(987)

def coin_change_limited(coins, amount, limits):
    dp = [0] + [float('inf')] * amount
    parent = [None] * (amount + 1)
    
    for i, coin in enumerate(coins):
        for x in range(amount, coin - 1, -1):
            for k in range(1, limits[i] + 1):
                if x >= k * coin:
                    if dp[x - k * coin] + k < dp[x]:
                        dp[x] = dp[x - k * coin] + k
                        parent[x] = (x - k * coin, coin, k)
    
    # Reconstruct the solution
    coin_count = {coin: 0 for coin in coins}
    current = amount
    
    while parent[current] is not None:
        prev, coin, count = parent[current]
        coin_count[coin] += count
        current = prev
    
    return coin_count if dp[amount] != float('inf') else None

if __name__=="__main__":
    coins = [500, 100, 50, 10, 5, 1]
    limits = [1, 3, 1, 5, 5, 10]  # Corresponding limits for each coin
    amount = 987
    result = coin_change_limited(coins, amount, limits)
    
    if result:
        print(f"金額: {amount}円の最小硬貨枚数の組合わせ:")
        for coin in coins:
            if result[coin] > 0:
                print(f"{coin}円: {result[coin]}枚")
        total_coins = sum(result.values())
        print(f"合計枚数: {total_coins}枚")
    else:
        print("解がありません")    