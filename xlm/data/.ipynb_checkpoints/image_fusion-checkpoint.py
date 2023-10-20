import pkuseg
import numpy as np
import fastBPE
import torch
import random
import pickle

NEVER_SPLIT_TAG = ['<s>', '</s>', '<pad>', '<unk>', '<special0>', '<special1>', '<special2>', '<special3>',
                   '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>', '<special10>']


class ImageFusion(object):

    def __init__(self, lang1, params):
        self.lang1 = lang1
        self.params = params
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.lang1_id = params.lang2id[self.lang1]
        self.image_id = params.image_id
        self.codes_path = params.codes_path
        self.vocab_path = params.vocab_path
        self.bpe = fastBPE.fastBPE(self.codes_path, self.vocab_path)
        self.image_embs = self.load_image_embs()
#         self.segment_vocab = list(self.image_embs.keys()) + NEVER_SPLIT_TAG
#         self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=NEVER_SPLIT_TAG)
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=NEVER_SPLIT_TAG)
        self.special_tags = set(NEVER_SPLIT_TAG)
        self.image_bpe = True # 爬取图片时是否是token级别的爬取 token/word


    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.pad_index)  # [sen_length * batch_size]

        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    # 得到该语言的image表示{"token":[emb1, emb2, emb3]}, emb是tensor.half类型
    def load_image_embs(self):
        final_en_image_embs = {}

        if self.params.best_image:
            with open(self.lang1+"_best_image.pickle", "rb") as f_en:
                en_embs_dict = pickle.load(f_en)
                final_en_image_embs = en_embs_dict
        
        elif self.params.best_top2_image:
            with open(self.lang1+"_best_top2_image.pickle", "rb") as f_en:
                en_embs_dict = pickle.load(f_en)
            final_en_image_embs = en_embs_dict
        
        else:
            with open("../../TCMA/clip/CLIP/"+self.lang1+"_image_embs_dict_cpu.pickle", "rb") as f_en:
                en_embs_dict = pickle.load(f_en)
            f_en_id = open("../../TCMA/"+self.lang1+"id2token_image.txt", "r", encoding="utf-8")
            en_id2token = {}
            while True:
                line = f_en_id.readline()
                if not line:
                    break
                en_id2token[line.strip().split("\t")[1]] = line.split("\t")[0]
            f_en_not_rgb_list = open("../../TCMA/clip/CLIP/"+self.lang1+"_not_rgb_list_cpu", "r", encoding="utf-8")
            en_not_rgb_list = set()
            while True:
                line = f_en_not_rgb_list.readline()
                if not line:
                    break
                en_not_rgb_list.add(line.strip())
            final_en_image_embs = {}
            for en_key in en_embs_dict.keys():
                if en_key in en_not_rgb_list:
                    continue
                # id = en_id2token[en_key.split("/")[0].split("_")[1]]
                id = en_id2token[en_key.split("/")[1].split("_")[1]]
                if id in final_en_image_embs.keys():
                    final_en_image_embs[id].append(en_embs_dict[en_key].float())
                else:
                    final_en_image_embs[id] = [en_embs_dict[en_key].float()]
        return final_en_image_embs

    
    def convert_to_text(self, batch, lengths, dico):
        """
        Convert a batch of sentences to a list of text sentences.
        """
        batch = batch.cpu().numpy()
        lengths = lengths.cpu().numpy()

        slen, bs = batch.shape
        assert lengths.max() == slen and lengths.shape[0] == bs
        assert (batch[0] == self.eos_index).sum() == bs
        assert (batch == self.eos_index).sum() == 2 * bs
        sentences = []

        for j in range(bs):
            words = ["</s>"]
            for k in range(1, lengths[j]):
                if batch[k, j] == self.eos_index:
                    break
                words.append(dico[batch[k, j]])
            words.append("</s>")
            sentences.append(words)
        return sentences
    
    def convert_to_text_bpe(self, batch, lengths, dico):
        batch = batch.cpu().numpy()
        lengths = lengths.cpu().numpy()

        slen, bs = batch.shape
#         assert lengths.max() == slen and lengths.shape[0] == bs
#         assert (batch[0] == self.eos_index).sum() == bs
#         assert (batch == self.eos_index).sum() == 2 * bs
        sentences = []

        for j in range(bs):
            words = ["</s>"]
            for k in range(1, lengths[j]):
                if batch[k, j] == self.eos_index:
                    break
                words.append(dico[batch[k, j]])
            words.append("</s>")
            sentences.append(words)
        return sentences
    
    def add_image_relational_bpe(self, batch, lengths, dico):
#         print("len_batch", batch.shape)
        split_sent_batch = self.convert_to_text(batch, lengths, dico)
#         print(split_sent_batch)
#         print("len_split_sent_batch", len(split_sent_batch))
#       batch_size * [每个句子插入图片的位置/图片表示]
        all_image_token_pos_in_one_batch = []
        all_image_in_one_batch = []
        for split_sent in split_sent_batch:
            cur_id = -1
            all_image_in_one_sent = []
            all_image_token_pos_in_one_sent = []
            # create tree
            for token in split_sent:
                cur_id += 1
                if token in self.special_tags:
                    continue
                elif token in self.image_embs.keys():
                    all_image_token_pos_in_one_sent.append(cur_id)
#                     if self.params.best_image:
#                         all_image_in_one_sent.extend(self.image_embs[token])
#                     if self.
                    all_image_in_one_sent.extend(random.sample(list(self.image_embs.get(token, [])), 1))
                else:
                    continue
            if len(all_image_in_one_sent) > 0:
                all_image_in_one_batch.append(torch.stack(all_image_in_one_sent, 0))
            else:
                all_image_in_one_batch.append(torch.zeros(1))
            all_image_token_pos_in_one_batch.append(torch.LongTensor(all_image_token_pos_in_one_sent))
#             print(len(all_image_token_pos_in_one_sent))
#             print(len(all_image_in_one_sent))
        return all_image_in_one_batch, all_image_token_pos_in_one_batch        
                    
        
    def add_image_embs_bpe(self, batch, lengths, dico, max_len=512):
        split_sent_batch = self.convert_to_text(batch, lengths, dico)
#         print(split_sent_batch)
#         print(stop)
        align_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
#         print(split_sent_batch)
        # 一个batch中所有句子中的图片tensor，batch_size * 每个句子中插入图片的个数
        # 如batch_size = 3 :[[tensor1, tensor2], [tensor3, tensor4, tensor5], []]
        # 可能出现图片embs的个数大于图片img的现象，因为超过512被切割了
        all_image_in_one_batch = []
        for split_sent in split_sent_batch:
            all_image_in_one_sent = []
            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:
                if token in self.special_tags:
                    words = []
                    token_bpe = [token]
                else:
                    words = [["<unk>"] for _ in range(min(len(list(self.image_embs.get(token, []))), int(self.params.max_align_image)))]
                    all_image_in_one_sent.extend(random.sample(list(self.image_embs.get(token, [])), min(len(list(self.image_embs.get(token, []))), int(self.params.max_align_image))))
                    token_bpe = [token]
                sent_tree.append((token_bpe, words))

                # if token[0] in self.special_tags:
                #     print("token in special")
                #     token_pos_idx = [pos_idx + 1]
                #     token_abs_idx = [abs_idx + 1]
                # else:
                token_pos_idx = [pos_idx + i for i in range(1, len(token_bpe) + 1)]
                token_abs_idx = [abs_idx + i for i in range(1, len(token_bpe) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in words:
                    # ent_bpe = self.bpe.apply(ent)[0].split(" ")

                    # ent_pos_idx = [token_pos_idx[-1] + 1]
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    # ent_abs_idx = [abs_idx + 1]
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
#             print(len(all_image_in_one_sent))
#             print(torch.stack(all_image_in_one_sent, 0).shape)
#           如果len为0说明这句话没有要插入的图片
            if len(all_image_in_one_sent) > 0:
                all_image_in_one_batch.append(torch.stack(all_image_in_one_sent, 0))
            else:
                all_image_in_one_batch.append(torch.zeros(1))
#             print("梯度", all_image_in_one_batch[0].requires_grad)

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word[0] in self.special_tags:
                    know_sent += word
                    seg += [self.lang1_id]
                else:
                    add_word = word
                    know_sent += add_word
                    seg += [self.lang1_id] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = sent_tree[i][1][j]
                    know_sent += add_word
                    seg += [self.image_id] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num), dtype=bool)
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = True
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = True
            align_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)


        index_list = []
        for sent in align_sent_batch:
            sent_id = []
            for word in sent:
                sent_id.append(dico.index(word, no_unk=False))
            sent_id = np.array(sent_id)
            index_list.append(sent_id)

        lengths = torch.LongTensor([len(s) for s in index_list])

        # x, sent_len = self.batch_sentences(index_list)
        # max_length = lengths.max().item() if lengths.max().item() < 512 else 512
        if lengths.max().item() <= max_len:
            max_length = lengths.max().item()
        else:
            lengths_mask = lengths > max_len
            lengths.masked_fill_(lengths_mask, max_len)
            max_length = max_len
        # max_length = max(all_len) if max(all_len) < 512 else 512

        final_align_sent_batch = []
        final_position_batch = []
        final_visible_matrix_batch = []
        final_seg_batch = []
        for i in range(len(index_list)):
            know_sent = list(index_list[i])
            pos = position_batch[i]
            seg = seg_batch[i]
            visible_matrix = visible_matrix_batch[i]
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [self.pad_index] * pad_num
                seg += [self.lang1_id] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((False, pad_num), (False, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                # know_sent[-1] = self.eos_index
                seg = seg[:max_length]
                # seg[-1] = self.lang1_id
                pos = pos[:max_length]
                # pos[-1] = max_length - 1
                visible_matrix = visible_matrix[:max_length, :max_length]
            final_align_sent_batch.append(know_sent)
            final_position_batch.append(pos)
            final_visible_matrix_batch.append(visible_matrix)
            final_seg_batch.append(seg)
            # 去掉多余的图片 保证图片个数和langs里面的一样多
            if max_length == max_len:
                image_count = 0
                for j in seg:
                    if j == self.image_id:
                        image_count += 1
                all_image_in_one_batch[i] = all_image_in_one_batch[i][:image_count]


#         print(all_image_in_one_batch[0].shape)
        sents_tensor = torch.LongTensor(final_align_sent_batch).t()
        pos_tensor = torch.LongTensor(final_position_batch).t()
        visible_matrix_tensor = torch.LongTensor(final_visible_matrix_batch)
        lang_tensor = torch.LongTensor(final_seg_batch).t()

        return sents_tensor, lengths, pos_tensor, visible_matrix_tensor, lang_tensor, all_image_in_one_batch


        

    def add_image_embs(self, split_sent_batch, dico):
        """
                input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
                return: know_sent_batch - list of sentences with image embedding 句子中图片的表示先用token/word表示 训练的时候会换成unk 然后换成对应的图片
                        position_batch - list of position index of each character.
                        visible_matrix_batch - list of visible matrixs
                        seg_batch - list of language/image tags
                """
        # 英中图片的爬取是token-level的 以后会是word_level
#         print(sent_batch)
#         split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
#         print(split_sent_batch)
        # token_level的需要bpe
        if self.image_bpe:
            all_bpe_sent_batch = []
            for sent in split_sent_batch:
                bpe_sent = []
                for token in sent:
                    if token in self.special_tags:
                        bpe_sent.append(token)
                    else:
                        bpe_sent.extend(self.bpe.apply([token])[0].split(" "))
                all_bpe_sent_batch.append(bpe_sent)
            split_sent_batch = all_bpe_sent_batch
#             split_sent_batch = [self.bpe.apply(sent).split(" ") for sent in split_sent_batch]
        # split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        # split_sent_batch = [sent.split(" ") for sent in sent_batch]
#         print(split_sent_batch)
        align_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
#         print(split_sent_batch)
        # 一个batch中所有句子中的图片tensor，batch_size * 每个句子中插入图片的个数
        # 如batch_size = 3 :[[tensor1, tensor2], [tensor3, tensor4, tensor5], []]
        # 可能出现图片embs的个数大于图片img的现象，因为超过512被切割了
        all_image_in_one_batch = []
        for split_sent in split_sent_batch:
            all_image_in_one_sent = []
            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:
                if token in self.special_tags:
                    words = []
                    token_bpe = [token]
                else:
                    words = [["<unk>"] for _ in range(min(len(list(self.image_embs.get(token, []))), int(self.params.max_align_image)))]
                    # all_image_in_one_sent.extend([token for _ in range(min(len(list(self.image_embs.get(token, []))), int(self.params.max_align_word)))])
                    all_image_in_one_sent.extend(random.sample(list(self.image_embs.get(token, [])), min(len(list(self.image_embs.get(token, []))), int(self.params.max_align_image))))
                    
                    
#                     print(all_image_in_one_sent)
                    # if len(list(self.image_embs[token])) <= int(self.params.max_align_word):
                    #     words = [token for i in range(len(list(self.image_embs[token])))]
                    # else:
                    #     words = [token for i in range(int(self.params.max_align_word))]
                        # words = random.sample(list(self.lookup_table.get(token, [])), self.params.max_align_word)

                    #                     words = list(self.lookup_table.get(token, []))[:self.params.max_align_word]
                    # words_bpe = [self.bpe.apply(word)[0].split(" ") for word in words]
                    token_bpe = self.bpe.apply([token])[0].split(" ")

                sent_tree.append((token_bpe, words))

                # if token[0] in self.special_tags:
                #     print("token in special")
                #     token_pos_idx = [pos_idx + 1]
                #     token_abs_idx = [abs_idx + 1]
                # else:
                token_pos_idx = [pos_idx + i for i in range(1, len(token_bpe) + 1)]
                token_abs_idx = [abs_idx + i for i in range(1, len(token_bpe) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in words:
                    # ent_bpe = self.bpe.apply(ent)[0].split(" ")

                    # ent_pos_idx = [token_pos_idx[-1] + 1]
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    # ent_abs_idx = [abs_idx + 1]
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
#             print(len(all_image_in_one_sent))
#             print(torch.stack(all_image_in_one_sent, 0).shape)
#           如果len为0说明这句话没有要插入的图片
            if len(all_image_in_one_sent) > 0:
                all_image_in_one_batch.append(torch.stack(all_image_in_one_sent, 0))
            else:
                all_image_in_one_batch.append(torch.zeros(1))
#             print("梯度", all_image_in_one_batch[0].requires_grad)

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word[0] in self.special_tags:
                    know_sent += word
                    seg += [self.lang1_id]
                else:
                    add_word = word
                    know_sent += add_word
                    seg += [self.lang1_id] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = sent_tree[i][1][j]
                    know_sent += add_word
                    seg += [self.image_id] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num), dtype=bool)
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = True
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = True
            align_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)


        index_list = []
        for sent in align_sent_batch:
            sent_id = []
            for word in sent:
                sent_id.append(dico.index(word, no_unk=False))
            sent_id = np.array(sent_id)
            index_list.append(sent_id)

        lengths = torch.LongTensor([len(s) for s in index_list])

        # x, sent_len = self.batch_sentences(index_list)
        # max_length = lengths.max().item() if lengths.max().item() < 512 else 512
        if lengths.max().item() <= 512:
            max_length = lengths.max().item()
        else:
            lengths_mask = lengths > 512
            lengths.masked_fill_(lengths_mask, 512)
            max_length = 512
        # max_length = max(all_len) if max(all_len) < 512 else 512

        final_align_sent_batch = []
        final_position_batch = []
        final_visible_matrix_batch = []
        final_seg_batch = []
        for i in range(len(index_list)):
            know_sent = list(index_list[i])
            pos = position_batch[i]
            seg = seg_batch[i]
            visible_matrix = visible_matrix_batch[i]
            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [self.pad_index] * pad_num
                seg += [self.lang1_id] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((False, pad_num), (False, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                # know_sent[-1] = self.eos_index
                seg = seg[:max_length]
                # seg[-1] = self.lang1_id
                pos = pos[:max_length]
                # pos[-1] = max_length - 1
                visible_matrix = visible_matrix[:max_length, :max_length]
            final_align_sent_batch.append(know_sent)
            final_position_batch.append(pos)
            final_visible_matrix_batch.append(visible_matrix)
            final_seg_batch.append(seg)
            # 去掉多余的图片 保证图片个数和langs里面的一样多
            if max_length == 512:
                image_count = 0
                for j in seg:
                    if j == self.image_id:
                        image_count += 1
                all_image_in_one_batch[i] = all_image_in_one_batch[i][:image_count]


#         print(all_image_in_one_batch[0].shape)
        sents_tensor = torch.LongTensor(final_align_sent_batch).t()
        pos_tensor = torch.LongTensor(final_position_batch).t()
        visible_matrix_tensor = torch.LongTensor(final_visible_matrix_batch)
        lang_tensor = torch.LongTensor(final_seg_batch).t()

        return sents_tensor, lengths, pos_tensor, visible_matrix_tensor, lang_tensor, all_image_in_one_batch


if __name__ == "__main__":
    # print("x")
    # for i in range(2):
    #     print(i)

    a = [torch.tensor([[1, 2, 3],[3, 4, 5], [111, 1112, 1123]]), torch.tensor([[7, 8, 9], [10, 11, 12]]), torch.tensor([[13, 14, 15], [16, 17, 18]])]
    x = torch.zeros(5, 3, dtype=a[0].dtype)
    idx = [torch.tensor([[0, 0, 0], [2, 2, 2], [3, 3, 3]])]
    x.scatter_(0, idx[0], a[0])
    print(x)
    print(a[0][0])
    image_or_not = a[0][0] == 3
    print(image_or_not)
    b = []
    for i in range(image_or_not.shape[0]):
        if image_or_not[i]:
            b.append(torch.tensor([i for _ in range(768)]))
    print(b)
    # print(a[0].device)
    # a = [i.to("cuda") for i in a]
    # # for i in a:
    # #     i.to("cuda:0")
    # #     print(i)
    # print(a)
    # print(a[0].device)
    # print(torch.has_cuda)


#     bpe = fastBPE.fastBPE(codes_path, vocab_path)
#     tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=segment_vocab)
#     a = ["我爱自然语言处理", "你怎么会机器翻译啊"]
#     split_sent_batch = [tokenizer.cut(sent) for sent in a]
#
#     split_sent_batch = [bpe.apply(sent)[0].split(" ") for sent in split_sent_batch]
