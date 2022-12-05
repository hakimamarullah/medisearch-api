from model.search_response import SearchResponse

def to_search_response(data):
    obj = {
        "doc_id" : data[0][0],
        "score": round(data[1],2),
        "contents": data[0][1]
    }

    return SearchResponse.parse_obj(obj)