from rest_framework.decorators import api_view
from rest_framework.response import Response
from utils.letor import Ranker

# Create your views here.
@api_view(['GET'])
def search(request):
    query = request.GET.get('q')
    result = Ranker.get_documents(query)
    return Response(data={result}, content_type='application/json')