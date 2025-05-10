
import pandas as pd
from langchain_core.documents import Document

path= r"D:\\Vaibhav_PC\\GenerativeAI\\GI\\ECommerce\\E-commerce_chatbot\\data\\flipkart_product_review.csv"
def dataconveter():
    product_data=pd.read_csv(path)

    data=product_data[["product_title","review"]]

    product_list = []

    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        # Construct an object with 'product_name' and 'review' attributes
        obj = {
                'product_name': row['product_title'],
                'review': row['review']
            }
        # Append the object to the list
        product_list.append(obj)

        
            
    docs = []
    for entry in product_list:
        metadata = {"product_name": entry['product_name']}
        doc = Document(page_content=entry['review'], metadata=metadata)
        docs.append(doc)
    # print(docs)
    return docs

if __name__=="__main__":
    dataconveter()