import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

import math




# 获取平均数 英文：average
# def get_average(list):
def get_average(num):
    sum = 0
    for i in range(len(num)):
        sum += num[i]
    return sum/len(num)
 
 
# 极差 英文：range
def get_range(num):
    return max(num) - min(num)
 
 
# 中位数 英文：median
def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2)-1
        return listnum[i]
    else:
        i = int(lnum / 2)-1
        return (listnum[i] + listnum[i + 1]) / 2
 
# 获取方差 英文：variance
def get_variance(num):
    sum = 0
    average = get_average(num)
    for i in range(len(num)):
        sum += (num[i] - average)**2
    return sum/len(num)
 
# 获取标准差 英文 standard deviation
def get_stddev(num):
    average = get_average(num)
    sdsq = sum( [(num[i] - average) ** 2 for i in range(len(num))] )
    stdev = (sdsq / (len(num) - 1)) ** .5
    return stdev
 
# 获取n阶原点距
def get_n_moment(num,n):
    sum = 0
    for i in ange(len(num)):
        sum += num[i]**n
    return sum/len(num)



## This is original softmax cross-entropy
class InnerProductWithScaleButNoUse(nn.Module):
    def __init__(self, feature_dim, class_num, scale=20):
        super(InnerProductWithScaleButNoUse, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)

        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))


        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################

        ############################## Theta ##############################
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
        
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        stdv_theta = get_stddev(thetas)
        print('Now stdv of thetas is {:.4f}'.format(stdv_theta))

        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))
        ############################## Theta ##############################


        # Calculate logits
        logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))

        return cos, logits, avg_theta, min_theta, max_theta, stdv_theta, avg_w_norm, avg_x_norm




class NormalizedInnerProductWithScale(nn.Module):
    """
    Paper:[COCOv2]
    Rethinking Feature Discrimination and Polymerization for Large scale recognition
    """
    def __init__(self, feature_dim, class_num, scale=20):
        super(NormalizedInnerProductWithScale, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)

        # print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format((sum(norm_weights)/len(norm_weights)).item(), (sum(norm_features)/len(norm_features)).item() ) )
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################

        ############################## Theta ##############################
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
        
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        stdv_theta = get_stddev(thetas)
        print('Now stdv of thetas is {:.4f}'.format(stdv_theta))

        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))
        ############################## Theta ##############################
        # Calculate logits
        logits = self.scale * cos

        return cos, logits, avg_theta, min_theta, max_theta, stdv_theta, avg_w_norm, avg_x_norm




class CosFaceInnerProduct(nn.Module):
    """
    Paper:[CosFace] and [AM-Softmax]
    CosFace: Large Margin Cosine Loss for Deep Face Recognition;
    Additive Margin Softmax for Face Verification.
    """
    def __init__(self, feature_dim, class_num, scale=20.0, margin=0.3):
        super(CosFaceInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables.scatter_(1, torch.unsqueeze(label, dim=-1), self.margin)
        # Calculate marginal logits
        marginal_logits = self.scale * (cos - margin_tables)

        return logits, marginal_logits
    



class ArcFaceInnerProduct(nn.Module):
    """
    Paper:[ArcFace]
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    """
    def __init__(self, feature_dim, class_num, scale=30.0, margin=0.5, easy_margin=False):
        super(ArcFaceInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # Setting behavior according to the value of easy_margin
        if easy_margin == -1: # If easy_margin = -1, just as the statement in the paper.
            self.Threshholder = -1
            self.out_indicator = 0
        elif easy_margin is True: # If easy_margin is True, for angle bigger than pi/2, do not add margin.
            self.Threshholder = 0
            self.out_indicator = 0
        else: # If easy_margin is Flase, do some adjustment for adding margin.
            self.Threshholder = - math.cos(self.margin)
            self.out_indicator = 1

    def forward(self, feat, label, return_ip=False):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################
        innerproduct_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables_ext = torch.zeros_like(cos)
        margin_tables_ext = Variable(margin_tables_ext).cuda()
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
    
        avg_theta = sum(thetas) / len(thetas)
        # avg_theta = mediannum(thetas)
        max_theta = max(thetas)
        min_theta = min(thetas) 
        stdv_theta = get_stddev(thetas)
        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))

        for i in range(margin_tables.size(0)):
            label_i = int(label[i])
            if cos[i, label_i].item() > self.Threshholder:
                margin_tables[i, label_i] += self.margin
            else:
                margin_tables_ext[i, label_i] -= self.margin * math.sin(self.margin)
        # Calculate marginal logits
        margin_tables_ext *= self.out_indicator
        marginal_logits = self.scale * (torch.cos(torch.acos(cos) + margin_tables) + margin_tables_ext)
        if return_ip:
            return cos, marginal_logits, innerproduct_logits
        else:
            return cos, marginal_logits



############################################################################

# DBM original
class DynamicBalancedMarginInnerProduct(nn.Module):
    def __init__(self, feature_dim, class_num, scale=30.0, easy_margin=False):
        super(DynamicBalancedMarginInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # Setting behavior according to the value of easy_margin
        if easy_margin == -1: # If easy_margin = -1, just as the statement in the paper.
            self.Threshholder = -1
            self.out_indicator = 0
        elif easy_margin is True: # If easy_margin is True, for angle bigger than pi/2, do not add margin.
            self.Threshholder = 0
            self.out_indicator = 0
        else: # If easy_margin is Flase, do some adjustment for adding margin.
            self.out_indicator = 1

    def forward(self, feat, label, return_ip=False):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################
        innerproduct_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables_ext = torch.zeros_like(cos)
        margin_tables_ext = Variable(margin_tables_ext).cuda()
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)

        max_theta = max(thetas)
        min_theta = min(thetas) 
        # avg_theta = max(mediannum(thetas), sum(thetas) / len(thetas), (max_theta + min_theta) / 2) # 3 in 1
        avg_theta = sum(thetas) / len(thetas)
        stdv_theta = get_stddev(thetas)
        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))


        for i in range(margin_tables.size(0)):
            label_i = int(label[i])
            if thetas[i] > avg_theta:
                if max_theta < 90:
                    the_margin = (90 - avg_theta) / 180 * math.pi
                else:
                    the_margin = 0
                if cos[i, label_i].item() > - math.cos(the_margin):
                    margin_tables[i, label_i] += the_margin
                else:
                    margin_tables_ext[i, label_i] -= the_margin * math.sin(the_margin)
            else:
                the_margin = min_theta / 180 * math.pi
                if cos[i, label_i].item() > - math.cos(the_margin):
                    margin_tables[i, label_i] += the_margin
                else:
                    margin_tables_ext[i, label_i] -= the_margin * math.sin(the_margin)
        # Calculate marginal logits
        margin_tables_ext *= self.out_indicator
        marginal_logits = self.scale * (torch.cos(torch.acos(cos) + margin_tables) + margin_tables_ext)
        if return_ip:
            return cos, marginal_logits, innerproduct_logits
        else:
            return cos, marginal_logits, avg_theta, min_theta, max_theta, stdv_theta, avg_w_norm, avg_x_norm




class NormalizedInnerProductWithAutoScale(nn.Module):
    def __init__(self, feature_dim, class_num, scale=20):
        super(NormalizedInnerProductWithAutoScale, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)

        # print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format((sum(norm_weights)/len(norm_weights)).item(), (sum(norm_features)/len(norm_features)).item() ) )
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################

        ############################## Theta ##############################
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
        
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        stdv_theta = get_stddev(thetas)
        print('Now stdv of thetas is {:.4f}'.format(stdv_theta))

        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))
        ############################## Theta ##############################
        # Calculate logits
        # logits = self.scale * cos

        # This is computing according to upper P
        # max_p = 0.999
        # s = math.log(max_p * (self.class_num - 1) / (1.0 - max_p))
        # This is computing according to range of P
        # r = 0.999
        # m = self.class_num - 1
        # s = math.log( (r * (m * m +1) + m * math.sqrt(r * r * (m * m - 2) + 4)) / (2 * m * (1 - r)) )
        
        # Fastest for pi/4
        s = math.sqrt(2.0) * math.log(self.class_num)
        logits = s * cos

        return cos, logits, avg_theta, min_theta, max_theta, stdv_theta, avg_w_norm, avg_x_norm




##################################################
class DBMwithAutoSInnerProduct(nn.Module):
    def __init__(self, feature_dim, class_num, scale=30.0, easy_margin=False):
        super(DBMwithAutoSInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # Setting behavior according to the value of easy_margin
        if easy_margin == -1: # If easy_margin = -1, just as the statement in the paper.
            self.Threshholder = -1
            self.out_indicator = 0
        elif easy_margin is True: # If easy_margin is True, for angle bigger than pi/2, do not add margin.
            self.Threshholder = 0
            self.out_indicator = 0
        else: # If easy_margin is Flase, do some adjustment for adding margin.
            self.out_indicator = 1

    def forward(self, feat, label, return_ip=False):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################
        innerproduct_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables_ext = torch.zeros_like(cos)
        margin_tables_ext = Variable(margin_tables_ext).cuda()
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)

        max_theta = max(thetas)
        min_theta = min(thetas) 
        # avg_theta = max(mediannum(thetas), sum(thetas) / len(thetas), (max_theta + min_theta) / 2) # 3 in 1
        avg_theta = sum(thetas) / len(thetas)
        stdv_theta = get_stddev(thetas)
        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))


        for i in range(margin_tables.size(0)):
            label_i = int(label[i])
            if thetas[i] > avg_theta:
                if max_theta < 90:
                    the_margin = (90 - avg_theta) / 180 * math.pi
                else:
                    the_margin = 0
                if cos[i, label_i].item() > - math.cos(the_margin):
                    margin_tables[i, label_i] += the_margin
                else:
                    margin_tables_ext[i, label_i] -= the_margin * math.sin(the_margin)
            else:
                the_margin = min_theta / 180 * math.pi
                if cos[i, label_i].item() > - math.cos(the_margin):
                    margin_tables[i, label_i] += the_margin
                else:
                    margin_tables_ext[i, label_i] -= the_margin * math.sin(the_margin)
        # Calculate marginal logits
        margin_tables_ext *= self.out_indicator
        s = math.sqrt(2.0) * math.log(self.class_num)
        marginal_logits = s * (torch.cos(torch.acos(cos) + margin_tables) + margin_tables_ext)
        if return_ip:
            return cos, marginal_logits, innerproduct_logits
        else:
            return cos, marginal_logits, avg_theta, min_theta, max_theta, stdv_theta, avg_w_norm, avg_x_norm





class PcheckNormalizedInnerProductWithAutoScaleDynamicMed(nn.Module):
    def __init__(self, feature_dim, class_num, scale=20):
        super(PcheckNormalizedInnerProductWithAutoScaleDynamicMed, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = math.sqrt(2.0) * math.log(self.class_num - 1)
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)

        # print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format((sum(norm_weights)/len(norm_weights)).item(), (sum(norm_features)/len(norm_features)).item() ) )
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        logits = self.scale * cos

        thetas = []
        probs = F.softmax(logits).detach().cpu().numpy()
        gt_probs = []
        Bs = []
        for i in range(cos.size(0)):
            # Theta
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
            # Prob
            gt_prob = probs[i, label_i]
            gt_probs.append(gt_prob)
            # B
            B = math.exp(logits[i, label_i].data[0]) * (1.0/gt_prob - 1.0)
            Bs.append(B)
        # Thetas
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        # med_theta = mediannum(theta)
        # Ps
        max_p = max(gt_probs)
        min_p = min(gt_probs)
        avg_p = get_average(gt_probs)
        # B_avg_thetas
        B_avg = math.acos(math.log(get_average(Bs) / (self.class_num - 1)) / self.scale) / math.pi * 180 # Degree
        
        med_theta = mediannum(thetas)
        med_theta_in_pi = (med_theta / 180.0) * math.pi
        if med_theta_in_pi > math.pi/4.0:
            med_theta_in_pi = math.pi/4.0
        self.scale = (1.0/math.cos(med_theta_in_pi)) * math.log(get_average(Bs))  
        
        ###
        print('The largest P and it\'s theta is {:.6f} and {:.6f}'.format(max_p, min_theta))
        print('The smallest P and it\s theta is {:.6f} and {:.6f}'.format(min_p, max_theta))
        print('The average P is {:.6f}'.format(avg_p))
        print('The medi is {:.6f}'.format(med_theta))
        print('B_degree is {:.4f}, scale is {:.4f}'.format(B_avg, self.scale))

        return cos, logits, avg_theta, med_theta, B_avg, self.scale, 0, 0




class PcheckNormalizedInnerProductWithReform(nn.Module):
    def __init__(self, feature_dim, class_num, scale=20):
        super(PcheckNormalizedInnerProductWithReform, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = math.sqrt(2.0) * math.log(self.class_num - 1)
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)

        # print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format((sum(norm_weights)/len(norm_weights)).item(), (sum(norm_features)/len(norm_features)).item() ) )
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        
        
        logits = self.scale * cos


        thetas = []
        probs = F.softmax(logits).detach().cpu().numpy()
        gt_probs = []
        Bs = []
        for i in range(cos.size(0)):
            # Theta
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
            # Prob
            gt_prob = probs[i, label_i]
            gt_probs.append(gt_prob)
            # B
            B = math.exp(logits[i, label_i].data[0]) * (1.0/gt_prob - 1.0)
            Bs.append(B)
        # Thetas
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        # Ps
        max_p = max(gt_probs)
        min_p = min(gt_probs)
        avg_p = get_average(gt_probs)
        # B_avg_thetas
        B_avg = math.acos(math.log(get_average(Bs) / (self.class_num - 1)) / self.scale) / math.pi * 180 # Degree
        
        med_theta = mediannum(thetas)
        med_theta_in_pi = (med_theta / 180.0) * math.pi
        if med_theta_in_pi > math.pi/4.0:
            med_theta_in_pi = math.pi/4.0
        self.scale = (1.0/math.cos(med_theta_in_pi)) * math.log(get_average(Bs))
        self.scale = 12.5
        

        # s = (1 / math.cos(med_theta_in_pi/2.0)) * math.log(self.class_num - 1)        
        
        
        ###
        print('The largest P and it\'s theta is {:.6f} and {:.6f}'.format(max_p, min_theta))
        print('The smallest P and it\s theta is {:.6f} and {:.6f}'.format(min_p, max_theta))
        print('The average P is {:.6f}'.format(avg_p))
        print('The medi is {:.6f}'.format(med_theta))
        print('B_degree is {:.4f}, scale is {:.4f}'.format(B_avg, self.scale))

        ########Here
        p_tables = torch.div(torch.clamp(cos.detach(), 0.000001, 1).detach(), F.softmax(logits).detach()).detach()


        final_probs = p_tables * F.softmax(logits)
        # return cos, logits, avg_p, min_p, max_p, B_avg, 0, 0
        # return cos, torch.log(final_probs), avg_p, min_p, max_p, B_avg, 0, 0
        return cos, final_probs, avg_p, min_p, max_p, B_avg, 0, 0



# class PcheckNormalizedInnerProductWithSimilarity(nn.Module):
#     def __init__(self, feature_dim, class_num, scale=20):
#         super(PcheckNormalizedInnerProductWithSimilarity, self).__init__()
#         self.feature_dim = feature_dim
#         self.class_num = class_num
#         self.scale = math.sqrt(2.0) * math.log(self.class_num - 1)
#         self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
#         nn.init.xavier_uniform_(self.weights)
#         self.weights.requires_grad = False

#     def forward(self, feat, label):
#         # Unit vector for features
#         norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
#         normalized_features = torch.div(feat, norm_features)
#         # Unit vector for weights
#         norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
#         normalized_weights = torch.div(self.weights, norm_weights)

#         # print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format((sum(norm_weights)/len(norm_weights)).item(), (sum(norm_features)/len(norm_features)).item() ) )
#         # Normalized inner product, or cosine
#         cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
#         #############################################
#         onehot_tables = torch.ones_like(cos)
#         balance_tables = torch.ones_like(cos) / (2.0 * self.class_num - 1.0)
#         #############################################
#         logits = self.scale * cos

#         thetas = []
#         probs = F.softmax(logits).detach().cpu().numpy()
#         gt_probs = []
#         Bs = []
#         for i in range(cos.size(0)):
#             # Theta
#             label_i = int(label[i])
#             theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
#             onehot_tables[i, label_i] = -1.0
#             balance_tables[i, label_i] = 1.0
#             thetas.append(theta)
#             # Prob
#             gt_prob = probs[i, label_i]
#             gt_probs.append(gt_prob)
#             # B
#             B = math.exp(logits[i, label_i].data[0]) * (1.0/gt_prob - 1.0)
#             Bs.append(B)
#         # Thetas
#         max_theta = max(thetas)
#         min_theta = min(thetas) 
#         avg_theta = get_average(thetas)
#         # Ps
#         max_p = max(gt_probs)
#         min_p = min(gt_probs)
#         avg_p = get_average(gt_probs)
#         # B_avg_thetas
#         B_avg = math.acos(math.log(get_average(Bs) / (self.class_num - 1)) / self.scale) / math.pi * 180 # Degree
        
#         med_theta = mediannum(thetas)
#         med_theta_in_pi = (med_theta / 180.0) * math.pi
#         if med_theta_in_pi > math.pi/4.0:
#             med_theta_in_pi = math.pi/4.0
#         self.scale = (1.0/math.cos(med_theta_in_pi)) * math.log(get_average(Bs))
#         self.scale = 12.5
        
        
        
#         ###
#         print('The largest P and it\'s theta is {:.6f} and {:.6f}'.format(max_p, min_theta))
#         print('The smallest P and it\s theta is {:.6f} and {:.6f}'.format(min_p, max_theta))
#         print('The average P is {:.6f}'.format(avg_p))
#         print('The medi is {:.6f}'.format(med_theta))
#         print('B_degree is {:.4f}, scale is {:.4f}'.format(B_avg, self.scale))

#         ########Here

#         similarities = cos
#         distances = 0.5 * torch.pow(similarities + onehot_tables, 2) * balance_tables
#         # return cos, logits, avg_p, min_p, max_p, B_avg, 0, 0
#         return cos, distances, avg_p, min_p, max_p, B_avg, 0, 0




class PcheckArcFaceInnerProduct(nn.Module):
    """
    Paper:[ArcFace]
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    """
    def __init__(self, feature_dim, class_num, scale=35.0, margin=0.5, easy_margin=False):
        super(PcheckArcFaceInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # Setting behavior according to the value of easy_margin
        if easy_margin == -1: # If easy_margin = -1, just as the statement in the paper.
            self.Threshholder = -1
            self.out_indicator = 0
        elif easy_margin is True: # If easy_margin is True, for angle bigger than pi/2, do not add margin.
            self.Threshholder = 0
            self.out_indicator = 0
        else: # If easy_margin is Flase, do some adjustment for adding margin.
            self.Threshholder = - math.cos(self.margin)
            self.out_indicator = 1

    def forward(self, feat, label, return_ip=False):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        innerproduct_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables_ext = torch.zeros_like(cos)
        margin_tables_ext = Variable(margin_tables_ext).cuda()
        
        for i in range(margin_tables.size(0)):
            label_i = int(label[i])
            if cos[i, label_i].item() > self.Threshholder:
                margin_tables[i, label_i] += self.margin
            else:
                margin_tables_ext[i, label_i] -= self.margin * math.sin(self.margin)
        # Calculate marginal logits
        margin_tables_ext *= self.out_indicator
        marginal_logits = self.scale * (torch.cos(torch.acos(cos) + margin_tables) + margin_tables_ext)

        thetas = []
        probs = F.softmax(logits).detach().cpu().numpy()
        gt_probs = []
        Bs = []
        for i in range(cos.size(0)):
            # Theta
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
            # Prob
            gt_prob = probs[i, label_i]
            gt_probs.append(gt_prob)
            # B
            B = math.exp(logits[i, label_i].data[0]) * (1.0/gt_prob - 1.0)
            Bs.append(B)
        # Thetas
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        med_theta = mediannum(thetas)
        # Ps
        max_p = max(gt_probs)
        min_p = min(gt_probs)
        avg_p = get_average(gt_probs)
        # B_avg_thetas
        B_avg = math.acos(math.log(get_average(Bs) / (self.class_num - 1)) / self.scale) / math.pi * 180 # Degree
        print('The largest P and it\'s theta is {:.6f} and {:.6f}'.format(max_p, min_theta))
        print('The smallest P and it\s theta is {:.6f} and {:.6f}'.format(min_p, max_theta))
        print('The average P is {:.6f}'.format(avg_p))
        print('B is {:.4f}'.format(B_avg))

        if return_ip:
            return cos, marginal_logits, innerproduct_logits
        else:
            # return cos, marginal_logits, avg_theta, med_theta, max_p, B_avg, 0, 0
            return cos, marginal_logits, avg_theta, med_theta, B_avg, self.scale, 0, 0

