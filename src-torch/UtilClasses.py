import config
import torch
from tqdm import tqdm
from torch.cuda import is_available
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
device = torch.device("cuda" if is_available() else "cpu")

#---------------------------------#
import nltk
nltk.download("wordnet")
#---------------------------------#

class Trainer:
    def __init__(self, encoder, decoder, optimizer, scheduler, batchqueue, batch_sz, epochs, logging, ckpt, cov=1.0):
        self.ckpt_num = ckpt
        self.epochs = epochs
        self.cov_lambda = cov
        self.encoder = encoder
        self.decoder = decoder
        self.logging = logging
        self.batch_sz = batch_sz
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batchqueue = batchqueue
                   
    def __train_step(self, batch):
        self.optimizer.zero_grad()
        coverage, context_vector = 0, None, None
        hidden, state = self.encoder(batch.code, batch.ast)
        # initially feed in batch_sz tensor of start tokens <S>
        # for us start token is always first word in the vocab
        dec_input = torch.Tensor([config.BOS] * self.batch_sz).long().to(device)
        dec_input = dec_input.unsqueeze(1)

        step_losses, comment_batch = [], []
        dec_lens = list(batch.nl.size())[1]
        for i in range(1, dec_lens):
            final_dist, attn_dist, state, coverage, context_vector = self.decoder(dec_input, hidden,
                                                                                  prev_state=state,
                                                                                  coverage=coverage,
                                                                                  context_vector=context_vector,
                                                                                  code_ext=batch.code_ex,
                                                                                  max_oovs=len(batch.code_oovs))
            comment_batch.append(torch.argmax(final_dist, 1))
            dec_input = batch.nl[:, i].unsqueeze(1)
            actual_word = batch.nl_ex[:, i]
            gold_probs = torch.gather(final_dist, 1, actual_word.unsqueeze(1)).squeeze(1)
            cov_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_loss = -torch.log(gold_probs + 1e-12) + (self.cov_lambda * cov_loss)
            step_losses.append(step_loss)

        comment_batch = torch.transpose(torch.stack(comment_batch), 0, 1)
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens
        loss = torch.mean(batch_avg_loss)
        loss.backward()
        self.optimizer.step()
        return loss.item(), comment_batch


    def train(self):
        self.__retrieve_checkpoint()
        for epoch in range(self.epochs):
            temp_batchqueue = self.batchqueue.batcher(self.batch_sz)
            self.nl_list = self.batchqueue.nl_list
            
            for (nbatch, batch) in enumerate(tqdm(temp_batchqueue)):
                target_comment = batch.nl[:, 1:]
                loss, comment_batch = self.__train_step(batch)
                out = f"\n[INFO] Step: [{epoch}, {nbatch}] | Loss: {loss:.2f}\n"
                out += self.__get_result(comment_batch.detach().cpu().numpy(), 
                                         target_comment.detach().cpu().numpy(), 
                                         batch.code_oovs)
                
                if not nbatch % self.logging:
                    print(out, flush=True)
                if self.ckpt_num and nbatch and not nbatch % self.ckpt_num:
                    self.__store_checkpoint()
            self.scheduler.step()
            
    def __store_checkpoint(self):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, "torch_checkpoints/ckpt.pt")


    def __retrieve_checkpoint(self):
        try:
            checkpoint = torch.load("torch_checkpoints/ckpt.pt")
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            print("[INFO] Read checkpoint")
        except FileNotFoundError:
            print("[INFO] Starting from scratch")
          
            
    def __trim_string(self, s):
        trimmed = []
        for w in s:
            if w == "</S>" or w == "<PAD>":
                break
            trimmed.append(w)
        if not len(trimmed):
            trimmed.append("<PAD>")
            trimmed.append("<PAD>") # method4 requires hyp to be >1
        return trimmed


    def __get_result(self, comment_batch, target_comment, oovs, print_idx=None):
        """
        BLEU Scoring
        """
        batch_len = len(comment_batch)
        BLEU4_score, METEOR_score = 0, 0
        
        for i in range(batch_len):
            target = ["<PAD>" if not j else self.nl_list[j-1] for j in target_comment[i]]
            
            comment_hat = []
            for word in comment_batch[i]:
                if not word:
                    comment_hat.append("<PAD>")
                elif word <= config.vocab_size_nl :
                    comment_hat.append(self.nl_list[word-1])
                else:
                    comment_hat.append(oovs[word-1-config.vocab_size_nl])
            
            target = self.__trim_string(target)
            comment_hat = self.__trim_string(comment_hat)
            
            if print_idx is not None and print_idx == i:
                print(f"Prediction: {' '.join(comment_hat)}")
                print(f"Ground Truth: {' '.join(target)}", flush=True)
            
            smoothie = SmoothingFunction().method4
            BLEU4_score += sentence_bleu([target], 
                                         comment_hat, 
                                         auto_reweigh=True,
                                         smoothing_function=smoothie) * 100
                                         
            METEOR_score += meteor_score([' '.join(target)],
                                         ' '.join(comment_hat)) * 100
   
        return f"Cumulative BLEU4 score: {(BLEU4_score/batch_len):.2f}\n" + \
               f"Cumulative METEOR score: {(METEOR_score/batch_len):.2f}"
               
#-------------------------------------------------------------------------------------------------------#               
               
class Tester:
    def __init__(self, encoder, decoder, batchqueue, batch_sz):
        self.encoder = encoder
        self.decoder = decoder
        self.batch_sz = batch_sz
        self.batchqueue = batchqueue
                                              

    def __test_step(self, batch):
        comment_batch, coverage, context_vector = [], None, None
        hidden, state = self.encoder(batch.code, batch.ast)
        # initially feed in batch_sz tensor of start tokens <S>
        # for us start token is always first word in z, axis=1)
        dec_input = torch.Tensor([config.BOS] * self.batch_sz).to(device).long()
        dec_input = dec_input.unsqueeze(1)
        
        comment_batch = []
        for i in range(1, batch.nl.shape[1]):
            final_dist, _, state, coverage, context_vector = self.decoder(dec_input, hidden,
                                                                          prev_state=state,
                                                                          coverage=coverage,
                                                                          context_vector=context_vector,
                                                                          code_ext=batch.code_ex,
                                                                          max_oovs=len(batch.code_oovs))
            
            max_prob_words = torch.argmax(final_dist, dim=1)
            comment_batch.append(max_prob_words)
            max_prob_words = max_prob_words.detach().cpu().numpy()
            max_prob_words = [config.UNK if w >= config.vocab_size_nl else w for w in max_prob_words]
            dec_input = torch.Tensor(max_prob_words).to(device).long()
            dec_input = dec_input.unsqueeze(1)
        
        comment_batch = torch.transpose(torch.stack(comment_batch), 0, 1)
        return comment_batch


    def test(self):
        self.__retrieve_checkpoint()
        temp_batchqueue = self.batchqueue.batcher(self.batch_sz)
        self.nl_list = self.batchqueue.nl_list
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            for batch in tqdm(temp_batchqueue):
                target_comment = batch.nl[:, 1:]
                comment_batch = self.__test_step(batch)
                out = self.__get_result(comment_batch.detach().cpu().numpy(), 
                                        target_comment.detach().cpu().numpy(), 
                                        batch.code_oovs,
                                        print_idx=10)
                print(out, flush=True)


    def __retrieve_checkpoint(self):
        try:
            checkpoint = torch.load("torch_checkpoints/ckpt.pt")
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            print("[INFO] Read checkpoint")
        except FileNotFoundError:
            print("[INFO] Starting from scratch")
    
    
    def __trim_string(self, s):
        trimmed = []
        for w in s:
            if w == "</S>" or w == "<PAD>":
                break
            trimmed.append(w)
        if not len(trimmed):
            trimmed.append("<PAD>")
            trimmed.append("<PAD>") # method4 requires hyp to be >1
        return trimmed


    def __get_result(self, comment_batch, target_comment, oovs, print_idx=None):
        """
        BLEU Scoring
        """
        batch_len = len(comment_batch)
        BLEU4_score, METEOR_score = 0, 0
        
        for i in range(batch_len):
            target = ["<PAD>" if not j else self.nl_list[j-1] for j in target_comment[i]]
            
            comment_hat = []
            for word in comment_batch[i]:
                if not word:
                    comment_hat.append("<PAD>")
                elif word <= config.vocab_size_nl :
                    comment_hat.append(self.nl_list[word-1])
                else:
                    comment_hat.append(oovs[word-1-config.vocab_size_nl])
            
            target = self.__trim_string(target)
            comment_hat = self.__trim_string(comment_hat)
            
            if print_idx is not None and print_idx == i:
                print(f"Prediction: {' '.join(comment_hat)}")
                print(f"Ground Truth: {' '.join(target)}", flush=True)
            
            smoothie = SmoothingFunction().method4
            BLEU4_score += sentence_bleu([target], 
                                         comment_hat, 
                                         auto_reweigh=True,
                                         smoothing_function=smoothie) * 100
                                         
            METEOR_score += meteor_score([' '.join(target)], 
                                         ' '.join(comment_hat)) * 100
   
        return f"Cumulative BLEU4 score: {(BLEU4_score/batch_len):.2f}\n" + \
               f"Cumulative METEOR score: {(METEOR_score/batch_len):.2f}\n"