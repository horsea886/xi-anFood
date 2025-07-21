from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import warnings
import random, math, os
import json
from pathlib import Path

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['USER_DATA_PATH'] = './data/users.json'

# 全局变量存储数据
trn_user_items = None
store_id_to_name = None
stores_df = None
root_path = './data/ml-1m/'
item_sim_matrix = None  # 新增：物品相似度矩阵

# 职业映射字典
occupations = {
    0: "其他", 1: "学术/教育", 2: "艺术家", 3: "文员/行政",
    4: "大学生", 5: "客户服务", 6: "医生/医疗", 7: "管理",
    8: "农民", 9: "家庭主妇", 10: "学生", 11: "律师",
    12: "程序员", 13: "退休", 14: "销售/营销", 15: "科学家",
    16: "个体经营", 17: "技术员", 18: "工匠", 19: "失业"
}

# 自定义JSON编码器处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

def get_data(root_path):
    # 读取评分数据
    rnames = ['user_id', 'store_id', 'rating', 'timestamp']
    try:
        ratings = pd.read_csv(os.path.join(root_path, 'ratings.dat'), sep='::',
                              engine='python', names=rnames, encoding='utf-8-sig',
                              dtype={'user_id': 'int32', 'store_id': 'int32',
                                     'rating': 'float32', 'timestamp': 'int32'})
    except FileNotFoundError:
        print("ratings.dat文件未找到，请确保文件路径正确")
        return {}, {}, pd.DataFrame()

    # 确保没有NaN值
    ratings = ratings.fillna(0)
    ratings['user_id'] = ratings['user_id'].astype('int32')
    ratings['store_id'] = ratings['store_id'].astype('int32')

    # 读取店铺信息 - 处理可能的缺失值
    mnames = ['store_id', 'title', 'genres', 'avg_rating', 'per_con', 'num_cmt']
    try:
        stores = pd.read_csv(os.path.join(root_path, 'stores.dat'), sep='::',
                             engine='python', names=mnames, encoding='utf-8-sig',
                             dtype={'store_id': 'int32', 'title': 'str', 'genres': 'str',
                                    'avg_rating': 'float32', 'per_con': 'float32',
                                    'num_cmt': 'float32'})
    except FileNotFoundError:
        print("stores.dat文件未找到，请确保文件路径正确")
        return {}, {}, pd.DataFrame()

    # 处理可能的缺失值
    if 'num_cmt' in stores:
        stores['num_cmt'] = stores['num_cmt'].fillna(0).astype('int32')
    if 'per_con' in stores:
        stores['per_con'] = stores['per_con'].fillna(0)
    if 'avg_rating' in stores:
        stores['avg_rating'] = stores['avg_rating'].fillna(0)

    # 确保ID列是整数类型
    if 'store_id' in stores:
        stores['store_id'] = stores['store_id'].astype('int32')

    if not stores.empty:
        store_id_to_name = dict(zip(stores['store_id'], stores['title']))
    else:
        store_id_to_name = {}
        print("警告：店铺数据为空")

    # 使用全部数据作为训练集
    trn_user_items = {}
    if not ratings.empty:
        trn_data = ratings.groupby('user_id')['store_id'].apply(list).reset_index()
        for user, stores_list in zip(trn_data['user_id'], trn_data['store_id']):
            trn_user_items[user] = set(stores_list)
    else:
        print("警告：评分数据为空")

    return trn_user_items, store_id_to_name, stores

def get_user_data():
    """加载用户数据"""
    user_data_path = app.config['USER_DATA_PATH']
    try:
        if Path(user_data_path).exists():
            with open(user_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"加载用户数据时出错: {e}")
        return {}

def save_user_data(user_data):
    """保存用户数据"""
    try:
        with open(app.config['USER_DATA_PATH'], 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存用户数据时出错: {e}")

def create_new_user(user_id, user_info):
    """创建新用户并保存信息"""
    # 保存到用户数据库
    user_data = get_user_data()
    user_data[str(user_id)] = user_info
    save_user_data(user_data)
    return user_info

def get_user_info(user_id):
    """获取用户信息，如果不存在则返回None"""
    user_data = get_user_data()
    user_id_str = str(user_id)
    return user_data.get(user_id_str, None)

def build_item_sim_matrix(trn_user_items):
    '''构建物品相似度矩阵'''
    sim = {}
    num = {}  # 记录每个物品被多少用户交互过

    print('建立店铺相似度矩阵...')
    for uid, items in trn_user_items.items():
        for i in items:
            if i not in num:
                num[i] = 0
            num[i] += 1
            if i not in sim:
                sim[i] = {}
            for j in items:
                if j not in sim[i]:
                    sim[i][j] = 0
                if i != j:
                    sim[i][j] += 1

    print('计算店铺相似度...')
    for i, related_items in sim.items():
        for j, score in related_items.items():
            if i != j:
                sim[i][j] = score / math.sqrt(num[i] * num[j])

    return sim

def Item_CF_Recommend(trn_user_items, sim_matrix, target_user_id, K, N):
    '''
    trn_user_items: 表示训练数据，格式为：{user_id1: [store_id1, store_id2,...,store_idn], user_id2...}
    sim_matrix: 物品相似度矩阵
    target_user_id: 需要为其生成推荐的用户ID
    K: Ｋ表示每个历史物品选取的相似物品数量
    N: N表示给用户推荐的店铺数量
    '''
    # 检查目标用户是否在数据集中
    if target_user_id not in trn_user_items:
        return None

    # 获取用户历史交互过的物品
    hist_items = trn_user_items[target_user_id]

    # 初始化推荐物品得分
    items_rank = {}

    for item in hist_items:
        # 如果该物品在相似度矩阵中
        if item in sim_matrix:
            # 获取最相似的K个物品
            similar_items = sorted(sim_matrix[item].items(), key=lambda x: x[1], reverse=True)[:K]
            for similar_item, score in similar_items:
                # 排除用户已经交互过的物品
                if similar_item not in hist_items:
                    if similar_item not in items_rank:
                        items_rank[similar_item] = 0
                    items_rank[similar_item] += score
        else:
            # 如果物品不在相似度矩阵中，跳过
            continue

    # 如果没有找到合适的推荐物品
    if not items_rank:
        return None

    # 获得评分最高的N个物品
    top_items = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:N]
    top_items = [x[0] for x in top_items]

    return top_items

def user_cf_rec(trn_user_items, target_user_id, K, N):
    '''
    协同过滤推荐
    '''
    # 检查目标用户是否在数据集中
    if target_user_id not in trn_user_items:
        return None

    # 建立item->users倒排表
    item_users = {}
    for uid, items in trn_user_items.items():
        for item in items:
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(uid)

    # 计算用户协同过滤矩阵
    sim = {}
    num = {}
    for item, users in item_users.items():
        for u in users:
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for v in users:
                if u != v:
                    if v not in sim[u]:
                        sim[u][v] = 0
                    sim[u][v] += 1

    # 计算用户相似度矩阵
    for u, related_users in sim.items():
        for v, score in related_users.items():
            # 避免除以零错误
            if num[u] > 0 and num[v] > 0:
                sim[u][v] = score / math.sqrt(num[u] * num[v])

    # 为目标用户生成推荐
    items_rank = {}

    # 如果目标用户没有相似用户
    if target_user_id not in sim:
        return None

    # 获取最相似的K个用户
    similar_users = sorted(sim[target_user_id].items(), key=lambda x: x[1], reverse=True)[:K]

    # 计算推荐店铺的得分
    for v, score in similar_users:
        for item in trn_user_items[v]:
            if item not in trn_user_items[target_user_id]:
                if item not in items_rank:
                    items_rank[item] = 0
                items_rank[item] += score

    # 获取得分最高的N个店铺
    top_items = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:N]
    top_items = [int(x[0]) for x in top_items]  # 转换为整数

    return top_items

def hybrid_recommendation(trn_user_items, stores_df, user_id, algorithm='hybrid'):
    """
    优化后的混合推荐策略
    - 新用户画像推荐：使用多维度特征推荐
    - 老用户：协同过滤推荐
    """
    # 获取用户信息
    user_info = get_user_info(user_id)
    print(f"执行推荐 - 用户ID: {user_id}, 算法: {algorithm}, 用户信息: {user_info}")

    # 如果用户不存在，使用热门推荐
    if user_info is None:
        rec_type = "热门推荐"
        rec_items = get_top10_popular_stores()[:10]
        print(f"用户不存在 - 使用热门推荐")
        return [item['store_id'] for item in rec_items], rec_type, user_info

    # 新用户或没有历史行为的用户，使用改进的用户画像推荐
    if user_id not in trn_user_items or algorithm == 'profile':
        rec_type = "个性化推荐（基于用户画像）"
        rec_items = get_personalized_recommendation(user_info, stores_df)
        print(f"用户画像推荐结果: {rec_items}")
    else:
        # 老用户，根据选择的算法推荐
        if algorithm == 'user_cf':
            # 使用UserCF
            rec_items = user_cf_rec(trn_user_items, user_id, 80, 10)
            rec_type = "个性化推荐（基于用户的协同过滤）"
        elif algorithm == 'item_cf':
            # 使用ItemCF
            global item_sim_matrix
            if item_sim_matrix is None:
                item_sim_matrix = build_item_sim_matrix(trn_user_items)
            rec_items = Item_CF_Recommend(trn_user_items, item_sim_matrix, user_id, 80, 10)
            rec_type = "个性化推荐（基于店铺的协同过滤）"
        else:  # 默认混合推荐
            # 尝试使用ItemCF
            if item_sim_matrix is None:
                item_sim_matrix = build_item_sim_matrix(trn_user_items)
            rec_items = Item_CF_Recommend(trn_user_items, item_sim_matrix, user_id, 80, 10)
            if not rec_items:
                # 如果ItemCF失败，使用UserCF
                rec_items = user_cf_rec(trn_user_items, user_id, 80, 10)
                rec_type = "个性化推荐（基于用户的协同过滤）"
            else:
                rec_type = "个性化推荐（基于店铺的协同过滤）"

    # 如果推荐结果为空，则返回热门推荐
    if not rec_items:
        rec_type = "热门推荐"
        rec_items = get_top10_popular_stores()[:10]
        rec_items = [item['store_id'] for item in rec_items]
        print(f"无推荐结果 - 使用热门推荐")

    return rec_items, rec_type, user_info

def get_personalized_recommendation(user_info, stores_df, n=10):
    """
    优化后的用户画像推荐算法
    基于性别、年龄、职业多维度特征进行智能推荐
    避免返回固定店铺列表
    """
    # 如果店铺数据无效，返回空列表
    if stores_df is None or stores_df.empty:
        return []

    # 根据用户画像计算店铺权重
    user_profile_score = {}

    # 1. 基于性别偏好
    gender = user_info.get('gender', '')
    for idx, row in stores_df.iterrows():
        store_id = row['store_id']
        user_profile_score[store_id] = 0

        # 性别权重
        if gender == 'M':  # 男性
            if '火锅' in row['genres'] or '烧烤' in row['genres'] or '川菜' in row['title']:
                user_profile_score[store_id] += 3
            elif '甜点' in row['genres'] or '轻食' in row['title']:
                user_profile_score[store_id] -= 1
        else:  # 女性
            if '甜点' in row['genres'] or '轻食' in row['title'] or '日韩' in row['genres']:
                user_profile_score[store_id] += 3
            elif '火锅' in row['genres'] or '烧烤' in row['genres']:
                user_profile_score[store_id] -= 1

    # 2. 基于年龄段偏好
    age = user_info.get('age', 0)
    for idx, row in stores_df.iterrows():
        store_id = row['store_id']

        if 18 <= age <= 25:  # 年轻人群
            if '网红' in row['title'] or '学生' in row['title'] or '平价' in row['title']:
                user_profile_score[store_id] += 2
        elif 26 <= age <= 35:  # 上班族
            if '商务' in row['title'] or '聚会' in row['title'] or '环境' in row['title']:
                user_profile_score[store_id] += 2
        elif 36 <= age <= 45:  # 家庭人群
            if '家庭' in row['title'] or '亲子' in row['title'] or '安静' in row['title']:
                user_profile_score[store_id] += 2
        elif age >= 46:  # 中老年
            if '养生' in row['title'] or '清淡' in row['title'] or '滋补' in row['title']:
                user_profile_score[store_id] += 2

    # 3. 基于职业特征
    # 在原有代码基础上扩展职业特征推荐
    occupation = user_info.get('occupation', -1)
    for idx, row in stores_df.iterrows():
        store_id = row['store_id']

        # 职业特征推荐 - 扩展覆盖所有职业
        if occupation == 0:  # 其他
            if '特色' in row['title'] or '创意' in row['title'] or '新奇' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 2:  # 艺术家
            if '艺术' in row['title'] or '创意' in row['title'] or '展览' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 3:  # 文员/行政
            if '简餐' in row['genres'] or '午餐' in row['title'] or '套餐' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 5:  # 客户服务
            if '快餐' in row['genres'] or '工作餐' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 6:  # 医生/医疗
            if '健康' in row['title'] or '养生' in row['title'] or '清淡' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 8:  # 农民
            if '农家' in row['title'] or '土菜' in row['title'] or '实惠' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 9:  # 家庭主妇
            if '亲子' in row['title'] or '家庭' in row['title'] or '儿童' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 11:  # 律师
            if '商务' in row['title'] or '高档' in row['title'] or '包间' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 13:  # 退休
            if '养生' in row['title'] or '清淡' in row['title'] or '早茶' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation == 16:  # 个体经营
            if '平价' in row['title'] or '实惠' in row['title'] or '自助' in row['genres']:
                user_profile_score[store_id] += 1.5
        elif occupation == 18:  # 工匠
            if '手工' in row['title'] or '传统' in row['title'] or '小吃' in row['genres']:
                user_profile_score[store_id] += 1.5
        elif occupation == 19:  # 失业
            if '优惠' in row['title'] or '特价' in row['title'] or '平价' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation in [4, 10]:  # 大学生/学生
            if '平价' in row['title'] or '自助' in row['genres'] or '快餐' in row['genres']:
                user_profile_score[store_id] += 1.5
        elif occupation in [1, 15]:  # 学术/科学家
            if '安静' in row['title'] or '书吧' in row['title'] or '饮品' in row['genres']:
                user_profile_score[store_id] += 1.5
        elif occupation in [12, 17]:  # 程序员/技术员
            if '深夜' in row['title'] or '快餐' in row['genres'] or '外卖' in row['title']:
                user_profile_score[store_id] += 1.5
        elif occupation in [7, 14]:  # 管理/销售
            if '商务' in row['title'] or '高档' in row['title'] or '包间' in row['title']:
                user_profile_score[store_id] += 1.5

    # 4. 结合店铺质量评分
    for idx, row in stores_df.iterrows():
        store_id = row['store_id']
        # 评分标准化 (0-5分转为0-3分)
        rating_score = min(3, row.get('avg_rating', 0) / 5 * 3)
        # 评论数标准化 (0-1000条转为0-2分)
        comment_score = min(2, row.get('num_cmt', 0) / 500)

        # 总分为：用户画像分 + 质量分
        user_profile_score[store_id] += rating_score + comment_score

    # 5. 筛选和排序推荐结果
    # 转换为DataFrame便于处理
    scores_df = pd.DataFrame(
        list(user_profile_score.items()),
        columns=['store_id', 'score']
    )

    # 合并店铺信息
    scores_df = scores_df.merge(stores_df, on='store_id')

    # 排序并去除重复
    scores_df = scores_df.sort_values('score', ascending=False).drop_duplicates('store_id')

    # 处理低分店铺（可能没有匹配特征）
    avg_score = scores_df['score'].mean()
    if avg_score < 2:  # 如果平均分数过低
        # 使用加权算法重新排序
        scores_df['weighted_score'] = (
                scores_df['score'] * 0.5 +
                scores_df['avg_rating'] * 0.3 +
                scores_df['num_cmt'] * 0.2
        )
        scores_df = scores_df.sort_values('weighted_score', ascending=False)

    # 返回前n个店铺ID
    top_items = scores_df.head(n)['store_id'].tolist()

    return top_items

def get_top10_popular_stores():
    """找出评论数最多的店铺top10"""
    if stores_df is None or stores_df.empty:
        return []

    # 按评论数量降序排序
    popular_stores = stores_df.sort_values(by='num_cmt', ascending=False).head(10)
    return popular_stores[['store_id', 'title', 'genres', 'num_cmt', 'avg_rating', 'per_con']].to_dict(orient='records')

def get_top10_highest_rated_stores():
    """找出评分最高的店铺top10（至少需要10条评论）"""
    if stores_df is None or stores_df.empty:
        return []

    # 过滤有足够评论的店铺
    filtered = stores_df[stores_df['num_cmt'] >= 10]
    if filtered.empty:
        return []

    # 按评分降序排序
    top_rated = filtered.sort_values(by='avg_rating', ascending=False).head(10)
    return top_rated[['store_id', 'title', 'genres', 'avg_rating', 'num_cmt', 'per_con']].to_dict(orient='records')

@app.before_first_request
def initialize():
    global trn_user_items, store_id_to_name, stores_df
    trn_user_items, store_id_to_name, stores_df = get_data(root_path)

    # 确保stores_df正确处理
    if stores_df is not None and not stores_df.empty:
        if 'num_cmt' in stores_df:
            stores_df['num_cmt'] = stores_df['num_cmt'].fillna(0).astype('int32')
        if 'per_con' in stores_df:
            stores_df['per_con'] = stores_df['per_con'].fillna(0)
        if 'avg_rating' in stores_df:
            stores_df['avg_rating'] = stores_df['avg_rating'].fillna(0)
        if 'store_id' in stores_df:
            stores_df['store_id'] = stores_df['store_id'].astype('int32')
    else:
        print("警告：店铺数据初始化失败")

@app.route('/')
def index():
    # 默认加载最火爆榜单
    popular_stores = get_top10_popular_stores()

    # 将结果转为字典格式
    store_data = []
    for store in popular_stores:
        store_data.append({
            'title': store['title'],
            'genres': store['genres'],
            'num_cmt': store['num_cmt'],
            'avg_rating': store['avg_rating'],
            'per_con': store['per_con']
        })

    # 直接渲染模板并传递榜单数据
    return render_template('index.html',
                           leaderboard_data={
                               'title': '最火爆店铺TOP10（按评论数）',
                               'description': '根据店铺评论数量评选出的最受欢迎店铺',
                               'stores': store_data
                           })

@app.route('/login', methods=['POST'])
def login():
    try:
        user_id = int(request.form['user_id'])
    except ValueError:
        return jsonify({'error': '请输入有效的数字ID'})

    # 获取用户信息
    user_info = get_user_info(user_id)

    # 存储用户信息到session
    session['user_id'] = user_id
    if user_info:
        session['user_info'] = user_info
        return jsonify({
            'success': True,
            'user_exists': True,
            'user_id': user_id,
            'user_info': user_info
        })
    else:
        # 用户不存在，返回标志，要求前端注册
        return jsonify({
            'success': True,
            'user_exists': False,
            'user_id': user_id,
            'is_new_user': True
        })

@app.route('/save_user_info', methods=['POST'])
def save_user_info():
    try:
        data = request.get_json()
        user_id = int(data['user_id'])
        gender = data['gender']
        age = int(data['age'])
        occupation = int(data['occupation'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'无效的请求数据: {str(e)}'})

    # 验证数据
    if gender not in ['M', 'F']:
        return jsonify({'error': '无效的性别'})
    if age < 1 or age > 120:
        return jsonify({'error': '无效的年龄'})
    if occupation < 0 or occupation > 19:
        return jsonify({'error': '无效的职业'})

    # 创建用户信息字典
    user_info = {
        'gender': gender,
        'age': age,
        'occupation': occupation
    }

    # 创建新用户
    create_new_user(user_id, user_info)

    # 更新session
    session['user_info'] = user_info

    return jsonify({
        'success': True,
        'user_id': user_id,
        'user_info': user_info
    })

@app.route('/recommend', methods=['GET'])
def recommend():
    # 从session获取用户信息
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': '用户未登录'})

    # 获取算法参数，默认为混合推荐
    algorithm = request.args.get('algorithm', 'hybrid')  # 这里正确获取参数

    # 修复点：正确检查全局变量是否初始化
    if trn_user_items is None or store_id_to_name is None or stores_df is None:
        initialize()  # 重新初始化

        # 获取推荐
    try:
        rec_items, rec_type, user_info = hybrid_recommendation(
            trn_user_items, stores_df, user_id, algorithm
        )
    except Exception as e:
        print(f"推荐出错: {e}")
        # 出现错误时返回热门店铺
        rec_items = get_top10_popular_stores()[:10] if stores_df is not None else []
        rec_type = "热门推荐"
        user_info = get_user_info(user_id)

        # 修正点：确保 rec_items 是整数列表，不是其他数据类型
    if rec_items and isinstance(rec_items[0], dict):
        # 如果返回的是字典列表（如热门推荐数据结构），提取 store_id
        rec_items = [item['store_id'] for item in rec_items]

        # 转换店铺ID为名称
    store_names = []
    if store_id_to_name:
        for sid in rec_items:
            # 确保 sid 是可哈希类型（整数或字符串）
            try:
                # 尝试转换为整数
                sid_int = int(sid)
                name = store_id_to_name.get(sid_int, f"未知店铺（ID:{sid_int}）")
            except (ValueError, TypeError):
                # 无法转换则直接使用原始值
                name = f"无效店铺ID: {sid}"
            store_names.append(name)

    # 准备结果
    recommendations = []
    for idx, name in enumerate(store_names, 1):
        recommendations.append({
            'rank': idx,
            'name': name
        })

    return jsonify({
        'user_id': user_id,
        'rec_type': rec_type,
        'user_info': user_info,
        'recommendations': recommendations
    })

# 添加三个榜单的路由
@app.route('/top_popular')
def top_popular():
    stores = get_top10_popular_stores()
    return jsonify({
        'title': '最火爆店铺TOP10（按评论数）',
        'description': '根据店铺评论数量评选出的最受欢迎店铺',
        'stores': stores
    })

@app.route('/top_rated')
def top_rated():
    stores = get_top10_highest_rated_stores()
    return jsonify({
        'title': '评分最高店铺TOP10',
        'description': '用户评分最高的优质店铺（至少10条评论）',
        'stores': stores
    })

if __name__ == "__main__":
    # 确保用户数据目录存在
    Path('./data').mkdir(exist_ok=True, parents=True)
    app.run(debug=True, port=5001)