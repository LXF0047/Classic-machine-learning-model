import json
import random
import time
import pandas as pd
from tqdm import tqdm
import Levenshtein


def status(d):
    # 随机修改状态码
    flag = random.choice(['200', '302', '404'])
    if flag == '200':
        d['status_code'] = '200'
        d['status_msg'] = 'OK'
    elif flag == '302':
        d['status_code'] = '302'
        d['status_msg'] = 'Found'
    else:
        d['status_code'] = '404'
        d['status_msg'] = 'Not Found'
    return d


def random_datetime():
    # 生成随机日期时间  Wed, 15 Jan 2020 09:13:21
    h = str(random.randint(0, 23))
    m = str(random.randint(0, 59))
    s = str(random.randint(0, 59))
    if len(h) == 1:
        h = '0' + h
    if len(m) == 1:
        m = '0' + m
    if len(s) == 1:
        s = '0' + s
    hms = h + ':' + m + ':' + s
    date = pd.date_range('2019/1/1', '2020/8/1', freq='D')
    week = [int(i.strftime("%w")) for i in date]
    dataframe = pd.DataFrame({'date': date, 'week': week})
    week_dict = {'0': 'Sun', '1': 'Mon', '2': 'Tue', '3': 'Wed', '4': 'Thur', '5': 'Fri', '6': 'Sat'}
    month_dict = {'01': 'Jan','02': 'Feb','03': 'Mar','04': 'Apr','05': 'May','06': 'Jun','07': 'Jul','08': 'Aug',
                  '09': 'Sep','10': 'Oct','11': 'Nov','12': 'Dec'}
    dataframe['week'] = dataframe['week'].apply(lambda x: week_dict[str(x)] if str(x) in week_dict else 'xxx')
    dataframe['date'] = dataframe['date'].apply(lambda x: str(x).split('-'))
    dataframe['year'] = dataframe['date'].apply(lambda x: x[0])
    dataframe['month'] = dataframe['date'].apply(lambda x: x[1]).apply(lambda x: month_dict[str(x)] if str(x) in month_dict else 'xxx')
    dataframe['day'] = dataframe['date'].apply(lambda x: x[2].split()[0])
    dataframe['res'] = dataframe['week'] + ', ' + dataframe['day'] + ' ' + dataframe['month'] + ' ' + dataframe['year'] + ' ' + hms

    return random.choice(dataframe['res'].tolist())


def server_header_names():
    datetime = random_datetime()
    base = "DATE: %s GMT\nSERVER: Apache/2.4.6 (CentOS) PHP/7.2.26\nLAST-MODIFIED: Wed, 18 Dec 2019 22:16:09 GMT\nETAG: \"1cc8-59a01caa4dc40\"\nACCEPT-RANGES: bytes\nCONTENT-LENGTH: 7368\nCONTENT-TYPE: text/html; charset=UTF-8\n" % datetime
    base2 = "DATE: %s GMT\nCONTENT-LANGUAGE: en-US\nCONTENT-TYPE: text/html\nCONNECTION: close\nSERVER: JRun Web Server\n" % datetime

    return random.choice([base, base2])


def random_ip(t='src'):
    if t == 'src':
        part1 = 172
        part2 = 1
        part3 = random.randint(1, 200)
        part4 = random.randint(1, 200)
        return '%s.%s.%s.%s' % (str(part1), str(part2), str(part3), str(part4))
    else:
        part1 = 192
        part2 = 168
        part3 = random.randint(1, 200)
        part4 = random.randint(1, 200)
        return '%s.%s.%s.%s' % (str(part1), str(part2), str(part3), str(part4))


def interface():
    return random.choice(['em0', 'em1', 'em2', 'eth0', 'eth1', 'eth2'])


def adjust_dict():
    file_names = ['backup_1.txt', 'basic_1.txt', 'burden_1.txt', 'plugins_1.txt', 'themes_1.txt', 'thumb_1.txt',
                  'upload_1.txt']
    for i in file_names:
        with open('data/generate_wpscan/wp_dict/%s' % i, 'r') as r:
            content = r.readlines()
        with open('data/generate_wpscan/wp_dict/%s' % i, 'w') as w:
            for j in content:
                w.write(j.strip('wp-'))


def handel_data():
    demo = {
        "status_code": "200",
        "server_header_names": "DATE: Wed, 15 Jan 2020 09:13:21 GMT\nSERVER: Apache/2.4.6 (CentOS) PHP/7.2.26\nX-POWERED-BY: PHP/7.2.26\nLAST-MODIFIED: Tue, 14 Jan 2020 14:55:33 GMT\nETAG: \"9ffd5a9b601c56e4fa3616beca74503b\"\nLINK: <http://172.16.1.120/wp/index.php/wp-json/>; rel=\"https://api.w.org/\"\nCONTENT-LENGTH: 1688\nCONTENT-TYPE: application/rss+xml; charset=UTF-8\n",
        "event_source": "ratel_http",
        "interface": "",
        "dst_ip": "172.16.1.120",
        "src_ip": "10.211.55.13",
        "sensor_id": "eaa885e3bc26b0b3bc9f64e988c4b9ef",
        "uid": "135688662756463",
        "commid": "1:fFfGcSpZ12W8bV88/xVDtK7V0xE=",
        "host": "172.16.1.120",
        "sensor_ip": "172.16.1.103",
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53",
        "timestamp": "1599805794098",
        "method": "GET",
        "cookie_vars": "wordpress_test_cookie,wordpress_test_cookie",
        "uri": "/sdfsd/index.php/comments/feed/",
        "version": "HTTP/1.1",
        "source.type": "ratel_http",
        "src_port": "40448",
        "referrer": "http://172.16.1.120/wp/",
        "client_header_names": "HOST: 172.16.1.120\nACCEPT: */*\nACCEPT-ENCODING: gzip, deflate\nCOOKIE: wordpress_test_cookie=WP+Cookie+check; wordpress_test_cookie=WP+Cookie+check\nUSER-AGENT: Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_1 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D201 Safari/9537.53\nREFERER: http://172.16.1.120/wp/\n",
        "resp_body": "<?xml version=\"1.0\" encoding=\"UTF-8\"?><rss version=\"2.0\"\n\txmlns:content=\"http://purl.org/rss/1.0/modules/content/\"\n\txmlns:dc=\"http://purl.org/dc/elements/1.1/\"\n\txmlns:atom=\"http://www.w3.org/2005/Atom\"\n\txmlns:sy=\"http://purl.org/rss/1.0/modules/syndication/\"\n\t\n\t>\n<channel>\n\t<title>\n\tComments for wpscan\\xE5\\xB7\\xA5\\xE5\\x85\\xB7\\xE6\\xB5\\x8B\\xE8\\xAF\\x95\t</title>\n\t<atom:link href=\"http://172.16.1.120/wp/index.php/comments/feed/\" rel=\"self\" type=\"application/rss+xml\" />\n\t<link>http://172.16.1.120/wp</link>\n\t<description>Just another WordPress site</description>\n\t<lastBuildDate>Tue, 14 Jan 2020 14:55:33 +0000</lastBuildDate>\n\t<sy:updatePeriod>\n\thourly\t</sy:updatePeriod>\n\t<sy:updateFrequency>\n\t1\t</sy:updateFrequency>\n\t<generator>https://wordpress.org/?v=5.3.2</generator>\n\t\t\t<item>\n\t\t\t\t<title>\n\t\t\t\tComment on Hello world! by A WordPress Commenter\t\t\t\t</title>\n\t\t\t\t<link>http://172.16.1.120/wp/index.php/2020/01/14/hello-world/#comment-1</link>\n\t\t<dc:creator><![CDATA[A WordPress Commenter]]></dc:creator>\n\t\t<pubDate>Tue, 14 Jan 2020 14:55:33 +0000</pubDate>\n\t\t<guid isPermaLink=\"false\">http://172.16.1.120/wp/?p=1#comment-1</guid>\n\t\t\t\t\t<description><![CDATA[Hi, this is a comment.\nTo get started with moderating, editing, and deleting comments, please visit the Comments screen in the dashboard.\nCommenter avatars come from &lt;a href=&quot;https://gravatar.com&quot;&gt;Gravatar&lt;/a&gt;.]]></description>\n\t\t<content:encoded><![CDATA[<p>Hi, this is a comment.<br />\nTo get started with moderating, editing, and deleting comments, please visit the Comments screen in the dashboard.<br />\nCommenter avatars come from <a href=\"https://gravatar.com\">Gravatar</a>.</p>\n]]></content:encoded>\n\t\t\t\t\t\t</item>\n\t\t\t</channel>\n</rss>\n",
        "proto": "tcp",
        "dst_port": "80",
        "status_msg": "OK",
        "guid": "9e675f7c-76fd-48c3-8e74-5a936bd4d831",
        "collect_time": "1599805794148",
        "ts": "1579055758.170",
        "sensor_name": "ATD"
    }
    src_ip = set()
    while len(src_ip) < 25:
        src_ip.add(random_ip('src'))
    # dst_ip = set()
    # while len(dst_ip) < 2:
    #     dst_ip.add(random_ip('dst'))
    dst_ip = '192.168.1.23'

    file_names = ['backup_1.txt', 'basic_1.txt', 'plugins_1.txt', 'themes_1.txt', 'thumb_1.txt']

    total_uri = 0

    final_res = []
    for i in file_names:
        wp_uri = []
        with open('/home/lxf/projects/test/wpdict/%s' % i, 'r') as r:
            for j in r:
                wp_uri.append(j.split('\r\n'))
        total_uri += len(wp_uri)
        with open('/home/lxf/projects/test/wpdict/res/res.txt', 'a') as a:
            for uri in tqdm(wp_uri):
                for sip in src_ip:
                    demo['dst_ip'] = dst_ip
                    demo['host'] = sip
                    demo['referrer'] = 'http://%s/%s/' % (sip, random.choice(['index', 'demo', 'test', 'root', 'files']))
                    demo['server_header_names'] = server_header_names()
                    demo['interface'] = interface()
                    demo['src_ip'] = sip
                    demo['uri'] = uri
                    tmp_res = status(demo)
                    a.write(str(demo)+'\n')
                    final_res.append(tmp_res)

    print('字典中总uri数：%s，生成数据数：%s，应该生成数据数：%s' % (total_uri, len(final_res), total_uri*25*3))


def gen_dga():
    white_d = dict()
    print('white dga')
    with open('/home/lxf/projects/test/white_dga.txt', 'r') as r:
        for i in r:
            white_d[i.strip()] = 0
    # 'matsnu', 'suppobox', 'gozi'
    dga_uri = []
    print('读取准备写入的dga uri')
    for name in ['matsnu', 'suppobox', 'gozi']:
        with open('/data0/new_workspace/mlxtend_dga_bin_20190307/merge/demo/data/families/%s' % name, 'r') as r:
            num = 0
            for i in r:
                if 8 < len(i.split(',')[0].split('.')[0]) < 20:
                    for j in white_d:
                        if Levenshtein.distance(i.split('.')[0], j) < 3:
                            dga_uri.append(i.split(',')[0].strip())
                            num += 1
                            if num > 16700:
                                break
    print('单词型uri个数：', len(dga_uri))
    base = {'qclass_name': 'C_INTERNET', 'qtype_name': 'A', 'query': 'vcansj.com', 'event_source': 'ratel_dns',
            'trans_id': '37330', 'response_bytes': '0', 'interface': '', 'dst_ip':
                '202.96.128.86', 'source.type': 'ratel_dns', 'src_ip': '192.168.0.103', 'src_port': '59265',
            'sensor_id': 'eaa885e3bc26b0b3bc9f64e988c4b9ef', 'uid': '272099505674478', 'proto': 'udp',
            'commid': '1:iDa53YHxX4/JYsfkx7p24HcIKlQ=', 'dst_port': '53', 'guid': '8e769a4c-cdfc-4c1b-965e-033173382596',
            'collect_time': '1599979788737', 'sensor_ip': '172.16.1.103', 'request_bytes': '28', 'ts': '1573395047.912',
            'timestamp': '1599979788226', 'sensor_name': 'ATD'}

    with open('/home/lxf/projects/test/wpdict/res/dga_words_res.txt', 'w') as w:
        for i in dga_uri:
            base['query'] = i
            base['dst_ip'] = random_ip('dst')
            base['src_ip'] = random_ip('src')
            base['src_port'] = random.randint(59200, 59300)
            base['request_bytes'] = random.randint(20, 100)
            w.write(str(base) + '\n')



if __name__ == '__main__':
    gen_dga()

