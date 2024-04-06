
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
# import jieba
# import re
import streamlit as st
import pickle
from surprise import Reader, Dataset, SVD
# from scipy.sparse import csr_matrix
# from surprise.model_selection.validation import cross_validate


# STOP_WORD_FILE = 'vietnamese-stopwords.txt'
# with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
#     stop_words = file.read()

# stop_words = stop_words.split('\n')


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)
    
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
    
with open('index.pkl', 'rb') as f:
    index = pickle.load(f)
    
filename = "Products_ThoiTrangNam_raw.csv"
df = pd.read_csv(filename)

filename2 = "Products_ThoiTrangNam_rating_raw.csv"
df2 = pd.read_csv(filename2, sep="\t")



# Define a function to recommend products based on user input
def recommender(view_product,dictionary,tfidf,index):
    view_product=view_product.lower().split()
    
    kw_vector=dictionary.doc2bow(view_product)
    print("View product's vector :")
    print(kw_vector)
    sim = index[tfidf[kw_vector]]
    
    list_id=[]
    list_score=[]
    for i in range (len(sim)):
        list_id.append(i)
        list_score.append(sim[i])
        
    df_result=pd.DataFrame({'id':list_id,
                           'score':list_score})
    
    five_highest_score=df_result.sort_values(by="score",ascending=False).head(6)
    print("Five Highest Score:")
    print(five_highest_score)
    print("Ids to list")
    IdstoList=list(five_highest_score['id'])
    print(IdstoList)
    
    products_find=df[df.index.isin(IdstoList)]
    result=df[["product_id","product_name","image","price","rating"]]
    result=pd.concat([result,five_highest_score],axis=1).sort_values(by='score',ascending=False)

    return result

menu = ["Business Objective", "Content-based Filtering","Collaborative Filtering"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
  st.subheader("Business Objective")
  st.write ("""###### I.	Giới thiệu project 
●Recommender systems được sử dụng trong
nhiều lĩnh vực: tạo danh sách phát nhạc/video
cho các dịch vụ như Netflix, YouTube & Spotify,
đề xuất sản phẩm cho các dịch vụ như
Amazon, đề xuất nội dung cho các nền tảng
truyền thông xã hội (social media platform)
Facebook & Twitter. Những system có thể hoạt
động bằng cách sử dụng một single input (như
music), hay multiple input trong và trên các nền
tảng như news, books,... và truy vấn tìm kiếm
(search query).
●Recommender system là các thuật toán nhằm
đề xuất các item có liên quan cho người dùng
(Item có thể là phim để xem, văn bản để đọc, sản
phẩm cần mua hoặc bất kỳ thứ gì khác tùy thuộc
vào ngành dịch vụ).
●Recommender system thực sự quan trọng trong
một số lĩnh vực vì chúng có thể tạo ra một khoản
thu nhập khổng lồ hoặc cũng là một cách để nổi
bật đáng kể so với các đối thủ cạnh tranh.""")  
  st.image("recommend_sys.png") 
  st.write ("""###### II.	Bussiness Objective
●Shopee là một hệ sinh thái
thương mại “all in one”, trong
đó có shopee.vn, là một
website thương mại điện tử
đứng top 1 của Việt Nam và
khu vực Đông Nam Á.
● Trên trang này đã triển khai nhiều tiện ích hỗ trợ
nâng cao trải nghiệm người dùng và họ muốn xây
dựng nhiều tiện ích hơn nữa.
● Giả sử công ty này chưa triển khai Recommender
System và bạn được yêu cầu triển khai hệ thống
này, bạn sẽ làm gì?
""")
  st.markdown("[Link to  Shoppe app](https://shopee.vn/Th%E1%BB%9Di-Trang-Nam-cat.11035567/)")
  st.image("shoppe.jpg")







elif choice == 'Content-based Filtering':
    st.markdown("Content-based Filtering")
    
    type = st.radio("Filter data or Input data?", options=("Filter", "Input"))
    
    if type=="Input":
# Build the Streamlit app
        st.subheader("Content-based Filtering")
# Get user input query
        query = st.text_input("Enter a product name:")
       
# Recommend products based on the query
        if query:
            results = recommender(query,dictionary,tfidf,index)
            df_results1=results[:5].reset_index()
            df_results2=results[5:10].reset_index()
            st.write(df_results1)
            with st.container():
                col= st.columns(5)
                for i, row in df_results1.iterrows():
                    product_id = row["product_id"]
                    product_name = row['product_name']
                    product_price = row['price']
                    product_rating = row['rating']
                    product_image=row['image']
                    with col[i]:
                        st.subheader(f"ID: {product_id}")
                        st.image(product_image,caption="Hình ảnh mô tả sản phẩm")
                        st.write(f"**Product Name:** {product_name}")
                        if product_price is not None:
                          st.write(f"**Price:** {product_price:.2f} VNĐ")
                        if product_rating is not None:
                          st.write("**Rating**",product_rating)
            with st.container():
                col= st.columns(5)
                for i, row in df_results2.iterrows():
                    product_id = row["product_id"]
                    product_name = row['product_name']
                    product_price = row['price']
                    product_rating = row['rating']
                    product_image=row['image']
                    with col[i]:
                        st.subheader(f"ID: {product_id}")
                        st.image(product_image,caption="Hình ảnh mô tả sản phẩm")
                        st.write(f"**Product Name:** {product_name}")
                        if product_price is not None:
                          st.write(f"**Price:** {product_price:.2f} VNĐ")
                        if product_rating is not None:
                          st.write("**Rating**",product_rating)

        
    else:
      query_2=st.selectbox('Select :',df['product_name'].unique())
      if query_2:
        results = recommender(query_2,dictionary,tfidf,index)
        df_results1=results[:5].reset_index()
        df_results2=results[5:10].reset_index()
        st.write(df_results1)
        with st.container():
            col= st.columns(5)
            for i, row in df_results1.iterrows():
                product_id = row["product_id"]
                product_name = row['product_name']
                product_price = row['price']
                product_rating = row['rating']
                product_image=row['image']
                with col[i]:
                    st.subheader(f"ID: {product_id}")
                    st.image(product_image,caption="Hình ảnh mô tả sản phẩm")
                    st.write(f"**Product Name:** {product_name}")
                    if product_price is not None:
                      st.write(f"**Price:** {product_price:.2f} VNĐ")
                    if product_rating is not None:
                      st.write("**Rating**",product_rating)
        with st.container():
            col= st.columns(5)
            for i, row in df_results2.iterrows():
                product_id = row["product_id"]
                product_name = row['product_name']
                product_price = row['price']
                product_rating = row['rating']
                product_image=row['image']
                with col[i]:
                    st.subheader(f"ID: {product_id}")
                    st.image(product_image,caption="Hình ảnh mô tả sản phẩm")
                    st.write(f"**Product Name:** {product_name}")
                    if product_price is not None:
                      st.write(f"**Price:** {product_price:.2f} VNĐ")
                    if product_rating is not None:
                      st.write("**Rating**",product_rating)




elif choice == 'Collaborative Filtering':
    st.title("Collaborative Filtering Project")
    st.write(df2.head())
    df2['rating'] = df2['rating'].astype(float)
    
    user = st.selectbox('Select :',df2['user'].head(20).unique())
    id_u = df2[df2['user'] == user]['user_id'].iloc[0]
    df_score = df2[["product_id"]]
    df_score['EstimateScore'] = df_score['product_id'].apply(lambda x: model.predict(id_u, x).est) # est: get EstimateScore
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()
    data= pd.merge(df_score, df, on='product_id')
    data_end1=data[:5].reset_index()
    data_end2=data[5:10].reset_index()
    st.write(data_end1)
    with st.container():
        col= st.columns(5)
        for i, row in data_end1.iterrows():
            product_id = row["product_id"]
            product_name = row['product_name']
            product_price = row['price']
            product_rating = row['rating']
            product_image=row['image']
            with col[i]:
                st.subheader(f"ID: {product_id}")
                st.image(product_image,caption="Hình ảnh mô tả sản phẩm")
                st.write(f"**Product Name:** {product_name}")
                if product_price is not None:
                  st.write(f"**Price:** {product_price:.2f} VNĐ")
                if product_rating is not None:
                  st.write("**Rating**",product_rating)
    with st.container():
        col= st.columns(5)
        for i, row in data_end2.iterrows():
            product_id = row["product_id"]
            product_name = row['product_name']
            product_price = row['price']
            product_rating = row['rating']
            product_image=row['image']
            with col[i]:
                st.subheader(f"ID: {product_id}")
                st.image(product_image,caption="Hình ảnh mô tả sản phẩm")
                st.write(f"**Product Name:** {product_name}")
                if product_price is not None:
                  st.write(f"**Price:** {product_price:.2f} VNĐ")
                if product_rating is not None:
                  st.write("**Rating**",product_rating)





 









        
        

    


          

            






