from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.
@api_view(['GET'])
def search(request):
    return Response(data={"message": "success", "data":["halo", "world"]}, content_type='application/json')