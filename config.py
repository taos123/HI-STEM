# -*- coding: UTF-8 -*-

# Author : sun tao
# Time   : 2019/7/19
# E-mail : suntao@ict.ac.cn


class Config:

    # model parameters
    input_size = 250
    hidden_size = 300
    attention_size = 300
    class_number = 201

    data_path = "./../../data/"

    user_data_file = data_path + "gowalla_scopus_1104.dat"
    check_in_data_file = data_path + "Gowalla_totalCheckins.txt"
    check_in_embedding_data_file = data_path + 'gowalla_vector_new250d.dat'
    results_path = "./../../results/"
    model_name = "TUL_Attention_Gowalla.ckpt"
    results_file_name = results_path + "results.txt"

