import copy
import json
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util import Retry

class MyRequests:
    def __init__(
        self,
        timeout_connect = 10,
        timeout_read = 30,
        cnt_retry_request = 3,
        interval_requests_sec = 5):
        self._info_req_default = {
            "timeout_connect": timeout_connect,
            "timeout_read": timeout_read,
            "cnt_retry_request": cnt_retry_request,
            "interval_requests_sec": interval_requests_sec
        }
    
    def _generate_session(
        self,
        cnt_retry_request,
        interval_requests_sec):
        sess = requests.Session()
        retries = Retry(
            total = cnt_retry_request,
            backoff_factor = interval_requests_sec,
            status_forcelist = [500, 502, 503, 504])
        sess.mount("https://", HTTPAdapter(max_retries = retries))

    def _get_info_req(self, info_req_local):
        info_req = copy.deepcopy(self._info_req_default)
        if info_req_local is None:
            return info_req
        for key in info_req_local:
            if key in self._info_req_default:
                info_req[key] = info_req_local[key]
        return info_req
    
    def get(self, url, headers, info_req_local = None):
        '''
        - Args
            - url:str: 
            - headers:dict: 
            - info_req_local:dict: 
        '''
        info_req = self._get_info_req(info_req_local)
        # TODO: get()呼び出す側でtry-catch囲んでもよいが，こちらで囲んで例外返すようにすべき？
        sess = self._generate_session(
            info_req["cnt_retry_request"],
            info_req["interval_requests_sec"])
        return sess.get(
            url,
            headers = headers,
            timeout = (info_req["timeout_connect"], info_req["timeout_read"]))
    
    
    def post(self, url, headers, payload, info_req_local = None):
        '''
        - Args
            - url:str: 
            - headers:dict: 
            - payload:dict: 
            - info_req_local:dict: 
        '''
        info_req = self._get_info_req(info_req_local)
        # TODO: get()呼び出す側でtry-catch囲んでもよいが，こちらで囲んで例外返すようにすべき？
        sess = self._generate_session(
            info_req["cnt_retry_request"],
            info_req["interval_requests_sec"])
        return sess.post(
            url,
            headers = headers,
            data = payload, # WARNING: 今回使うとこではjson.dumps()するとエラる
            timeout = (info_req["timeout_connect"], info_req["timeout_read"]))
    
    def patch(self, url, headers, payload, info_req_local = None):
        '''
        - Args
            - url:str: 
            - headers:dict: 
            - payload:dict: 
            - info_req_local:dict: 
        '''
        info_req = self._get_info_req(info_req_local)
        # TODO: get()呼び出す側でtry-catch囲んでもよいが，こちらで囲んで例外返すようにすべき？
        sess = self._generate_session(
            info_req["cnt_retry_request"],
            info_req["interval_requests_sec"])
        return sess.patch(
            url,
            headers = headers,
            data = json.dumps(payload), # WARNING: 今回使うとこではjson.dumps()しないととエラる
            timeout = (info_req["timeout_connect"], info_req["timeout_read"]))