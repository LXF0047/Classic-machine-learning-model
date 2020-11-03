import numpy as np
import pandas as pd
from ETA.tls import Tls, ext, cs


class TorFeatureExtraction(object):
    def __init__(self):
        self.joy_compact = 1

    def get_feature(self, info):
        # info = pkts_intercept(info)
        cisco_joy_feature = list()
        cisco_joy_feature.extend(self._get_flow_packet_lengths(info))
        cisco_joy_feature.extend(self._get_tls_info(info))
        return cisco_joy_feature

    def _get_flow_packet_lengths(self, info):
        if self.joy_compact:
            num_rows = 10
            bin_size = 150.0
        else:
            num_rows = 60
            bin_size = 25.0
        trans_mat = np.zeros((num_rows, num_rows))
        packets_size = [float(x) for x in info["packets_size"].split(",")]
        if len(packets_size) <= 1:
            cur_size = min(int(packets_size[0] / bin_size), num_rows - 1)
            trans_mat[cur_size, cur_size] = 1
        else:
            for i in range(1, len(packets_size)):
                pre_size = min(int(packets_size[i-1]/bin_size), num_rows-1)
                cur_size = min(int(packets_size[i]/bin_size), num_rows-1)
                trans_mat[pre_size, cur_size] += 1
        # get empirical transition probabilities
        for i in range(num_rows):
            if float(np.sum(trans_mat[i:i + 1])) != 0:
                trans_mat[i:i + 1] = trans_mat[i:i + 1] / np.sum(trans_mat[i:i + 1])

        return list(trans_mat.flatten())

    def _get_tls_info(self, info):
        tls = Tls(info)
        cs_count = len(cs.keys())
        ext_count = len(ext.keys())
        s_ext_count = len(ext.keys())
        tls_info = np.zeros(cs_count + ext_count + s_ext_count + 7)
        for byte in tls.tls_ciphers2bytes(info.get("client_ciphers", "")):
            tls_info[cs[byte]] = 1

        extensions = info.get("client_extensions", "")
        s_extensions = info.get("extensions", "")
        server_name = info.get("server_name", "")
        server_name2 = self.server_name_2(server_name)
        subject = info.get('subject', "")
        issuer = info.get('issuer', "")
        subject_cn = ""
        try:
            items = [x.split("=") for x in subject.split(",")]
            for item in items:
                if len(item) >= 2 and item[0] == "CN":      # Common Name
                    subject_cn = item[1]
        except:
            subject_cn = ""
        issuer_cn = ""
        try:
            items = [x.split("=") for x in issuer.split(",")]
            for item in items:
                if len(item) >= 2 and item[0] == "CN":      # Common Name
                    issuer_cn = item[1]
        except:
            issuer_cn = ""
        free_cn = "Let's Encrypt Authority X3"
        SSL_server_cert_subna = info.get("SSL_server_cert_subna", "0")
        SSL_server_cert_days = info.get("SSL_server_cert_days", "0")
        client_key_exchange_length = info.get("client_key_exchange_length", "0")

        for extend in extensions.split(","):
            extend = extend.lower()
            if extend in ext:
                tls_info[cs_count + ext[extend]] = 1

        for extend in s_extensions.split(","):
            extend = extend.lower()
            if extend in ext:
                tls_info[cs_count + 9 + ext[extend]] = 1

        # joy中最后一位特征是c_key_length，但ratel中没有，所以先用tls的版本代替
        tls_info[cs_count + ext_count + s_ext_count] = tls.tls_version2code(info.get("version", "")) + \
                                                       tls.tls_version2code(info.get("client_version", ""))

        if SSL_server_cert_subna:
            tls_info[cs_count + ext_count + s_ext_count + 1] = int(SSL_server_cert_subna)
        if SSL_server_cert_days:
            tls_info[cs_count + ext_count + s_ext_count + 2] = int(SSL_server_cert_days)
        if client_key_exchange_length:
            tls_info[cs_count + ext_count + s_ext_count + 3] = int(client_key_exchange_length)
        # 自签名证书
        try:
            if (issuer_cn.split('.')[-1] == 'com') and (issuer_cn.split('.')[-3] == 'www'):
                tls_info[cs_count + ext_count + s_ext_count + 4] = 1
        except:
            tls_info[cs_count + ext_count + s_ext_count + 4] = 0
        try:
            if (subject_cn.split('.')[-1] == 'net') and (subject_cn.split('.')[-3] == 'www'):
                tls_info[cs_count + ext_count + s_ext_count + 5] = 1
        except:
            tls_info[cs_count + ext_count + s_ext_count + 5] = 0
        try:
            if server_name2.split('.')[-1] == 'onion':
                tls_info[cs_count + ext_count + s_ext_count + 6] = 1
        except:
            tls_info[cs_count + ext_count + s_ext_count + 6] = 0

        return tls_info

    @staticmethod
    def server_name_2(str1):
        str_list = str(str1).split('.')
        try:
            str_new = str_list[-2] + '.' + str_list[-1]
        except:
            str_new = str1
        return str_new

