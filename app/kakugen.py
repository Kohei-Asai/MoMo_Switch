import random

kakugen_list = [
    "人生はダイナミカルシステムだ.",
    "人生はトレードオフ.",
    "There is no free lunch.",
    "何かを得るためには、\n何かを差し出さねばならない(※人生の真実).",
    "もちろん天才は存分に活躍できます.",
    "関数は無限次元のベクトルだ！",
    "確率が収束してきた！アツい！",
    "ポインタって習いましたよね？",
]

def random_generate():
    index = random.randint(0, len(kakugen_list)-1)
    return kakugen_list[index]