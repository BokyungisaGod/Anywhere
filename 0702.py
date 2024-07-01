from langchain.schema import Document
from fastapi import APIRouter, Depends, HTTPException
from dependencies.database import provide_session
from domains.users.services import UserService
from domains.users.repositories import UserRepository
from domains.users.dto import (
    UserItemGetResponse,
    UserPostRequest,
    UserPostResponse,
    RequestDTO,
    ChainDTO,
    Location,
    KeyWordDataRequest, KeyWordDataResponse
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import bs4
from dependencies import d
from dependencies.kakaoapi import (search_places_nearby, get_coords_from_address)
from langchain_core.prompts import PromptTemplate

name = "users"
router = APIRouter()
prompt = PromptTemplate.from_template(
        """
        당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
        검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
        한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.


        #Question:
        {question}

        #Context:
        {context}

        #Answer:
        """
    )
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

class ChainHolder:
    def __init__(self):
        self.chain = None

chain_holder = ChainHolder()

def make_question(keywordInfor):
    try:
        print(f"Received search request for keyword information: {keywordInfor}")

        results = []

        for i in range(len(keywordInfor['place_name'])):
            name = keywordInfor['place_name'][i]
            latitude = keywordInfor['y'][i]
            longitude = keywordInfor['x'][i]

            # Find place_id by searching with the place name
            search_result = d.search_places(name)
            if not search_result:
                print(f"No search result found for place: {name}")
                continue

            place_id = search_result[0]['place_id']

            try:
                details = d.get_place_details(place_id)
                if details:
                    geometry = details.get('geometry', {})
                    location = geometry.get('location', {})
                    place_latitude = location.get('lat')
                    place_longitude = location.get('lng')

                    if d.compare_coordinates(latitude, longitude, place_latitude, place_longitude):
                        address = details.get('formatted_address', '주소 정보 없음')
                        rating = details.get('rating', '평점 정보 없음')

                        reviews = details.get('reviews', [])
                        review_list = []
                        if reviews:
                            for review in reviews:
                                author_name = review.get('author_name', '작성자 없음')
                                text = review.get('text', '리뷰 없음')
                                review_rating = review.get('rating', '평점 없음')
                                review_list.append({
                                    "author_name": author_name,
                                    "text": text,
                                    "rating": review_rating
                                })

                        results.append({
                            "name": name,
                            "address": address,
                            "rating": rating,
                            "reviews": review_list
                        })
                else:
                    print(f"No details found for place: {name}")
            except Exception as e:
                print(f"Error fetching details for place {name} with place_id {place_id}: {e}")

        print(f"Returning results: {results}")
        create_chain_review(results)  # Function call corrected
    except Exception as e:
        print(f"Error in make_question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing keyword information: {e}")


def create_chain_review(results):
    try:
        context = "\n".join([f"{res['name']} - {res['address']}, 평점: {res['rating']}" for res in results])
        documents = [Document(page_content=context, metadata={})]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30, separators=[". ", "."])
        splits = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        chain_holder.chain = rag_chain
        print("Chain has been set up.")
    except Exception as e:
        print(f"Error in create_chain_review: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating chain review: {e}")


def create_chain_web(web_paths):
    loader = WebBaseLoader(
        web_paths=[web_paths],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["article_head", "article_con"]},
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30, separators=[". ", "."])
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

@router.post("/getinfo")
async def getInfo(
    payload:KeyWordDataRequest
):
    keywordInfor = {}
    address = []
    x = payload.data.x
    y = payload.data.y
    distance = payload.data.distan
    keyword = payload.data.keyword
    keywordInfor, address = search_places_nearby(x,y,distance,keyword)
    if keywordInfor is not None:
        for i in range(len(address)):
            xy_list = get_coords_from_address(address[i])
            if xy_list is not None:
                keywordInfor['x'].append(xy_list[0])
                keywordInfor['y'].append(xy_list[1])
            else:
                keywordInfor['x'].append(None)
                keywordInfor['y'].append(None)
            print(xy_list)
        print(keywordInfor)
        make_question(keywordInfor)
        return KeyWordDataResponse(keywordinfo=keywordInfor)
    else:
        return None

@router.post(f"/{name}/create")
async def create(
    payload: UserPostRequest,
    db=Depends(provide_session),
) -> UserPostResponse:
    user_service = UserService(user_repository=UserRepository(session=db))

    user_id = user_service.create_user(
        user_name=payload.user_name,
        user_pw=payload.user_password,
    )

    return UserPostResponse(id=user_id).dict()

@router.get(f"/{name}/{{user_id}}")
async def get(
    user_id,
    db=Depends(provide_session),
) -> UserItemGetResponse:
    user_service = UserService(user_repository=UserRepository(session=db))

    user_info = user_service.get_user(user_id=user_id)

    return UserItemGetResponse(
        data=UserItemGetResponse.DTO(
            id=user_info.id,
            name=user_info.name,
            flavor_genre_first=user_info.flavor_genre_first,
            flavor_genre_second=user_info.flavor_genre_second,
            flavor_genre_third=user_info.flavor_genre_third,
            created_at=user_info.created_at,
            updated_at=user_info.updated_at,
        )
    ).dict()

@router.post("/chain")
async def set_chain(payload: ChainDTO):
    chain_holder.chain = create_chain_web(payload.web_paths)
    return {"message": "Chain has been set up."}

@router.post("/request")
async def process_question(payload: RequestDTO):
    if chain_holder.chain is None:
        raise HTTPException(status_code=400, detail="Chain is not set up.")
    
    # 질문에 대한 답변 가져오기
    question = payload.question
    answer = chain_holder.chain.invoke(question)
    if answer:
        return answer
    else:
        return "실행 오류"
