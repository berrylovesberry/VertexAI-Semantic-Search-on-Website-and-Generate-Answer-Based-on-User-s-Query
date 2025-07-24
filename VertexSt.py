from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import aiplatform
import requests, tempfile, pdfplumber
from bs4 import BeautifulSoup
from google.cloud import discoveryengine_v1beta
from google.cloud.discoveryengine_v1beta.types import SearchRequest
import streamlit as st

PROJECT_ID = "" # project created on on Google Cloud
LOCATION = ""  # eg "global"  
DATA_STORE_ID = "" # data store created on Google Cloud
SEARCH_APP_ID = "" # search app created on Google Cloud

# Construct the full search placement string
search_placement = f"projects/PROJECT_ID/locations/LOCATION/collections/default_collection/dataStores/DATA_STORE_ID/servingConfigs/default_search"
client = discoveryengine_v1beta.SearchServiceClient()
aiplatform.init(project="PROJECT_ID", location="LOCATION")

# --- LLM setup ---
# Option 1: Gemini (multimodal, recommended)
model = GenerativeModel("gemini-2.0-flash") 

# Option 2: PaLM (text-only)
# model = TextGenerationModel.from_pretrained("text-bison@002")

# --- Utility Functions ---
def vertex_search_top_links(query: str, top_k: int = 1):
    request = SearchRequest(
        serving_config=search_placement,
        query=query,
        page_size=top_k,
    )
    results = client.search(request)
    urls = []
    cnt = 0
    # get top 5 urls
    for result in results:
        if cnt >= 5:
            break
        cnt += 1
        doc = result.document
        derived = doc.derived_struct_data
        url = derived.get("formattedUrl", "")
        urls.append(url)

    return urls

def extract_text_from_pdf_url(pdf_url):
    r = requests.get(pdf_url, stream=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(r.content)
        tmp.flush()
        with pdfplumber.open(tmp.name) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

def extract_text_from_html_url(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup.get_text(separator='\n')

def get_text_from_url(url):
    try:
        return extract_text_from_pdf_url(url) if '.pdf' in url else extract_text_from_html_url(url)
    except Exception as e:
        print(f"❌ Failed to extract from {url}: {e}")
        return ""

def chunk_text(text, chunk_size=1500, overlap=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def summarize_with_vertex(chunk: str) -> str:
    # Gemini format
    response = model.generate_content(f"You are an IRS assistant, summarize the following IRS information:\n\n{chunk}")
    return response.text.strip()
    
# Or use PaLM instead:
# def summarize_with_vertex(chunk: str) -> str:
#     return model.predict(chunk, temperature=0.2, max_output_tokens=512).text.strip()

def summarize_query(urls: list):
    print("🔍 Extracting...")
    text = ""
    for url in urls:
        print(f"-> {url}")
        content = get_text_from_url(url)
        if content:
            text += "\n" + content

    chunks = chunk_text(text)
    print(f"🧩 {len(chunks)} chunks")

    print("🧠 Summarizing with Vertex AI...")
    summaries = [summarize_with_vertex(c) for c in chunks]  
    return "\n\n".join(summaries)

# --- Streamlit UI ---
st.set_page_config(page_title="My Tax Advisor") # layout="centered"
st.header("My Tax Advisor")
# 🧠 User prompt
query = st.text_area("Enter your prompt:") # 
st.markdown("Ask questions about tax rates, deductions, or any IRS-related topics. For example: 'What is the tax rate for short term and long term capital gain? Response is generated only based on irs.gov and other IRS official sites.")

# 🔍 Model call
if st.button("Generate Response"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating..."):
            try:
                top_urls = vertex_search_top_links(query)  
                summary = summarize_query(top_urls[0:5])
                response = model.generate_content(f"answer user's question {query} in a clear and informative way based on the following sources: {summary} and cite IRS links where helpful.")
                st.markdown("### Response:")
                st.write(response.text.strip())
            except Exception as e:
                st.error(f"Error: {e}")
