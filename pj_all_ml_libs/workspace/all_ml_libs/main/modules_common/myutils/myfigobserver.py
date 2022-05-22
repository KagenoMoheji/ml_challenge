import os
import matplotlib.pyplot as plt
import matplotlib.animation as pltanim
import pandas as pd

class FigObserver:
    '''
    matplotlibにおけるfigとそのsubplot(axes)の状態保持・監視するスーパークラス。
    plot処理はサブクラスにて関数を追加して実装。
    '''
    def __init__(self, figsize = (20, 20), cnt_row = 2, cnt_col = 2):
        '''
        - Args
            - figsize:tuple: 
            - cnt_row:int: サブプロットの行数
            - cnt_col:int: サブプロットの列数
        - Return
            - None
        '''
        self._fig = plt.figure(figsize = figsize)
        self._ax_matrix_params = [cnt_row, cnt_col]
        self._dict_ax = {}
        
    def __del__(self):
        '''
        "def <FigObserverのインスタンス>"された時の削除処理。
        '''
        # TODO: 複数のTextScatterインスタンスを生成していた時、一方をdelしたら他方のグラフも表示されなくなるのか？確認
        plt.cla()
        plt.clf()
        plt.close()
        
    def create_ax(self, ax_name, locate = 1, is_3d = False):
        '''
        サブプロットの作成。
        - Args
            - ax_name:str: サブプロットに割り当てる名称。
            - locate:int: figにおける何番目の配置か。
            - is_3d:bool: 3Dプロットか否か。
        - Returns
            - None(self._dict_axへの格納)
        
        TODO: タイトル・軸指定とかも実装
        '''
        projection = "3d" if is_3d else None
        self._dict_ax[ax_name] = {
            "subplot_params": "",
            "ax": self._fig.add_subplot(*self._ax_matrix_params, locate, projection = projection)
        }
        
    def delete_ax(self, ax_name):
        '''
        サブプロットの削除。
        - Args
            - ax_name:str: 削除するサブプロットに割り当てた名称。
        - Returns
            - None
        - Refs
            - https://qiita.com/KntKnk0328/items/5ef40d9e77308dd0d0a4#axes%E3%81%AE%E5%89%8A%E9%99%A4
        '''
        self._fig.delaxes(self._dict_ax[ax_name]["ax"])
        del self._dict_ax[ax_name]
    
    def print_ax_list(self):
        '''
        self._figが保持しているサブプロットの一覧表示。
        '''
        output = pd.DataFrame(
            {
                "ax_name": self._dict_ax.keys(),
                "subplot_params": [self._dict_ax[ax_name]["subplot_params"] for ax_name in self._dict_ax]
            }
        )
        print(output)
    
    def show(self):
        '''
        TODO: self._fig自身のみshowさせることはできないか…？plt.show()ではほかのfigもshowされてしまう。
        '''
        plt.show()
    
    def save(self, fname, save_params = {}):
        '''
        self._figのファイル出力。
        - Args
            - fname:str: 出力するファイルパス(絶対パス)。
        - Returns
            - None(ファイル出力)
        '''
        os.makedirs("/".join(fname.split("/")[:-1]) + "/", exist_ok = True)
        save_params.setdefault("facecolor", "#fff")
        save_params.setdefault("bbox_inches", None) # bbox_inches = "tight"
        self._fig.savefig(fname, **save_params)
    
    
#     def save_3d_animation(self, ax_name, fname):
#         '''
#         TODO: 実装失敗。理由はpltanim.FuncAnimation()のinit_func引数で描写処理を渡せていないため。self._figに施した描写処理を元にして欲しいのだが…
#         '''
#         def __init_func():
#             ???
        
#         def __animate(i):
#             self._dict_ax[ax_name].view_init(elev = 30., azim = 3.6 * i)
#             return self._fig,
        
#         anim = pltanim.FuncAnimation(self._fig, __animate, init_func = __init_func, frames = 100, interval = 100, blit = True)
#         anim.save(
#             fname,
#             writer = "ffmpeg", # "imagemagick" # "pillow"
#             dpi = 100)
        
