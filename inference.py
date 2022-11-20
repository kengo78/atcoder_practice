class BaseFilter:
    
    """stop_word_patternはクラス変数として保持"""
    stop_word_pattern = re.compile(r'はじめに|まとめ|目次|関連|最後に|お知らせ|新着|おわり|終わりに|さいご|もくじ|\
                                       カテゴリ|QA|Q&A|おまけ|メニュー|問合わせ|問い合わせ|問合せ|フォーム|\
                                       その他|コメント|申込|質問|口コミ|一覧|アクセス|スポンサーリンク|トピックス|プロフィール|\
                                       お申し込み|資料請求|ラインナップ|イベント|トピックス|\
                                       応募|体験記|採用情報|企業|検索|商品説明|ストア内|商品説明|無料サポート|\
                                       送信|ナビゲーション|コース|再販|募集概要|福利厚生|\
                                       連絡先|トップページ|プラン|注文|購入|その他|事業内容|募集|リンク|窓口|品揃え|料金表|概要|\
                                       ⑴|店長|配送|出荷|キャンペーン|記事|タグ|ボクシル')
    
    
    def __init__(self):
        """self.resultにはフィルター関数を全て実行した後の見出しのリストがインスタンス変数として格納される"""
        self.result = []
    
    
    def _valid(self, keyword, header_list=None, n=20):
        """ フィルター関数を全て実行した後の見出しのリストを返す

        Paramaters:
        -----------
        keyword : str
            キーワード
        header_list : List[str]
            見出し候補のリスト
        n : int
            関連キーワードを含む見出しを抽出する際に、上位n個までの関連キーワードを抽出できるよう指定

        Returns:
        -----------
        self.result : List[str]
            フィルター関数を全て実行した後の見出しのリスト
        """
        try:
            self.result = self.remove_stop_word(header_list)
            self.result = self.drop_duplicate(self.result)
            self.result = self.extract_top_n_related_word(keyword, n, self.result)
        except Exception as e:
            raise e
        
        return self.result
    
    
    @classmethod
    def remove_stop_word(cls, header_list=None):
        """ 要素のうちストップワードを含むものを除去した見出し候補のリストを返す

        Paramaters:
        -----------
        header_list : List[str]
            見出し候補のリスト

        Returns:
        -----------
        result_header_list : List[str]
            ストップワード除去後の見出し候補のリスト
        """
        
        result_header_list = []
        
        for _, header in enumerate(header_list):
            if not cls.stop_word_pattern.search(header):
                result_header_list.append(header)
            
        return result_header_list
    
    
    @staticmethod
    def drop_duplicate(header_list=None):
        """ 重複している要素を除去した見出し候補のリストを返す

        Paramaters:
        -----------
        header_list : List[str]
            見出し候補のリスト

        Returns:
        -----------
        result_header_list : List[str]
            重複要素除去後の見出し候補のリスト
        """
        
        result_header_list = list(set(header_list))
        
        return result_header_list
    
    
    @staticmethod
    def extract_top_n_related_word(keyword, n, header_list=None):
        """ 関連キーワード上位n番目見出し候補のリストを返す

        Paramaters:
        -----------
        keyword : str
            キーワード
        n : int
            関連キーワードを含む見出しを抽出する際に、上位n個までの関連キーワードを抽出できるよう指定
        header_list : List[str]
            見出し候補のリスト

        Returns:
        -----------
        result_header_list : List[str]
            重複要素除去後の見出し候補のリスト
        """

        result_header_list = []

        connection = pymysql.connect(host='staging-emma-aurora-cluster.cluster-cwhyhk8dpxoh.ap-northeast-1.rds.amazonaws.com',
                                     user='exiuser',
                                     password='dbpw3676',
                                     db='emma',
                                     charset='utf8',
                                     connect_timeout=1000,
                                     cursorclass=pymysql.cursors.DictCursor)
        
        with closing(connection.cursor()) as cursor:
            sql_1 = """select id from keywords
                    where keywords.name = %(keyword)s"""
            cursor.execute(sql_1, {"keyword": keyword.split()[0]})
            query_result_1 = cursor.fetchall()
            keyword_id = query_result_1[0].get("id")
            
            sql_2 = """select name from score_keywords
                    where score_keywords.keyword_id = %(keyword_id)s
                    and score_keywords.importance <= %(importance)s
                    and score_keywords.importance >= 1
                    ORDER BY importance ASC"""
            cursor.execute(sql_2, {"keyword_id": keyword_id, "importance": n})
            query_result_2 = cursor.fetchall()
            top_n_related_word_df = pandas.DataFrame(query_result_2)
            top_n_related_word_list = list(top_n_related_word_df.iloc[:, 0])
        
        for _, header in enumerate(header_list):
            for _, related_word in enumerate(top_n_related_word_list):
                related_word = related_word.split(" ")[1]
                related_word_pattern = re.compile(related_word)
                if related_word_pattern.search(header):
                    result_header_list.append(header)
                    break
                else:
                    pass
                
        
        return result_header_list
    
    
    @staticmethod
    def calc_bert_score(cands=None, refs=None):
        """ BERTスコアの算出

        Paramaters:
        -----------
        cands ([List[str]]): [比較元の文]
        refs ([List[str]]): [比較対象の文]

        Returns:
        -----------
            [(List[float], List[float], List[float])]: [(Precision, Recall, F1スコア)]
        """

        Precision, Recall, F1 = score(cands, refs, lang="ja", verbose=True)

        return Precision.numpy().tolist(), Recall.numpy().tolist(), F1.numpy().tolist()

    
    @staticmethod
    def evaluate(keyword, df_all, threshold, eval_index="F1", header_list=None):
        """ 評価指標を指定して、threshold(閾値)以上の評価値をもつ見出しのリストを返す

        Paramaters:
        -----------
        keyword : str
            キーワード
        header_list : List[str]
            見出し候補のリスト
        df_all : pandas.DataFrame
            学習データを全てマージしたデータフレーム
        threshold : float
            閾値
        eval_index : str
            評価指標(Precision,Recall,F1から選択)
            デフォルトは"F1"
                
        Returns:
        -----------
            List[str]
                評価後の見出しのリスト
        """
        
        ref_df = df_all[df_all["name"].str.contains(keyword)]
        
        ref_list = []
        for column in ref_df.columns:
            result.extend(ref_df[column].tolist())
        ref_list = [x for x in ref_list if pandas.isnull(x) == False]
        ref_list = list(set(ref_list))
        
        P_list = []
        R_list = []
        F1_list = []
        for header in self.header_list:
            P, R, F1 = self.calc_bert_score([header]*len(ref_list), ref_list)
            P_list.append(numpy.mean(P))
            R_list.append(numpy.mean(R))
            F1_list.append(numpy.mean(F1))

        score_df = pandas.DataFrame(columns=["header", "Precision", "Recall", "F1"])
        score_df["header"] = header_list
        score_df["Precision"] = P_list
        score_df["Recall"] = R_list
        score_df["F1"] = F1_list
        score_df = score_df.sort_values(eval_index, ascending=False)

        return score_df[score_df[eval_index] > threshold]['header'].to_list()

    
def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    return model, tokenizer


def separate_sentence(sentence):
    """文章を見出しの要素に分解してリストで返す

    Paramaters:
    -----------
    sentence : str
        モデルの出力(見出しが文字列で連結されている)

    Returns:
    -----------
    heading_list : List[str]
        見出しのリスト
    """

    output = sentence.split('[SEP]</s>')[1].replace('<s>', '').replace('<unk>', '')
    heading_list = output.split('[SEP]')[:-1]
    heading_list = [heading.replace(" ", "") for heading in heading_list]

    return heading_list


def predict_fn(data, model_and_tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = model_and_tokenizer
    model.eval()
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        model.cuda()
        
    data = data.pop("keyword", data)
    keywords = "".join(data.split())
    input_text = '<s>'+keywords+'[SEP]'
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(
        input_ids=input_ids,
        num_return_sequences=3,
        max_length=100,
        min_length=100,
        do_sample=True,
        top_k=40,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bad_word_ids=[[tokenizer.unk_token_id]]
    )
    
    heading_list = []
    for index, sentence in enumerate(tokenizer.batch_decode(output)):
        temp_list = separate_sentence(sentence)
        heading_list.extend(temp_list)

    heading_list = list(set(heading_list))
    
    # # filtering instance
    # filtering = BaseFilter()
    # # filtering process
    # heading_list = filtering.remove_stop_word(header_list=heading_list)
    # heading_list = filtering.drop_duplicate(header_list=heading_list)

    response = {"headings": []}
    for _, heading in enumerate(heading_list):
        heading_dict = {"tag": "h2",
                        "heading": heading}
        response["headings"].append(heading_dict)

    return response