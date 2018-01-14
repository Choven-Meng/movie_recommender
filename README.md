一、下载数据集
MovieLens 1M 数据集，包含6000个用户在近4000部电影上的1亿条评论。
1、用户数据(users.dat)
1::F::1::10::48067                                          
2::M::56::16::70072
3::M::25::15::55117
数据中的格式：UserID::Gender::Age::Occupation::Zip-code
分别对应：    用户ID  性别    年龄     职业ID    邮编
性别："M" for male and "F" for female
年龄： 1: "Under 18"    18: "18-24"    25: "25-34"    35: "35-44"    45: "45-49"    50: "50-55"    56: "56+"
职业：0: "other" or not specified
     1: "academic/educator" —— 2: "artist" —— 3: "clerical/admin" —— 4: "college/grad student" —— 5: "customer service"
     6: "doctor/health care" —— 7: "executive/managerial" —— 8: "farmer" —— 9: "homemaker" —— 10: "K-12 student"
     11: "lawyer" —— 12: "programmer" —— 13: "retired" —— 14: "sales/marketing" —— 15: "scientist" 
     16: "self-employed" —— 17: "technician/engineer" —— 18: "tradesman/craftsman" —— 19: "unemployed" —— 20: "writer"
 
 2、电影数据(movies.dat)
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
数据中的格式：MovieID::Title::Genres
分别对应：     电影ID  电影名  电影风格
电影风格有：Action、Adventure、Animation、Children's、Comedy、Crime、Documentary、Dram、Fantasy、Film-Noir、Horror、Musical、Mystery、                Romance、Sci-Fi、Thriller、War、Western  共18种风格

3、评分数据(ratings.dat)
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
数据中的格式：UserID::MovieID::Rating::Timestamp
分别对应：    用户ID   电影ID   评分     时间戳
UserIDs range between 1 and 6040
MovieIDs range between 1 and 3952
Ratings are made on a 5-star scale (whole-star ratings only)
Timestamp is represented in seconds since the epoch as returned by time(2)
Each user has at least 20 ratings

二、处理数据
分析数据：
UserID、Occupation和MovieID不用变。
Gender字段：需要将‘F’和‘M’转换成0和1。
Age字段：要转成7个连续数字0~6。
Genres字段：是分类字段，要转成数字。首先将Genres中的类别转成字符串到数字的字典，然后再将每个电影的Genres字段转成数字列表，因为有些电影是多个Genres的组合。
Title字段：处理方式跟Genres字段一样，首先创建文本到数字的字典，然后将Title中的描述转成数字的列表。另外Title中的年份也需要去掉。
Genres和Title字段需要将长度统一，这样在神经网络中方便处理。空白部分用‘< PAD >’对应的数字填充。
