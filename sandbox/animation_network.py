import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import copy


sample_num = 100

A_csv = 'A.csv'
B_csv = 'B.csv'
C_csv = 'C.csv'
D_csv = 'D.csv'
E_csv = 'E.csv'

A_com_csv = 'A_com.csv'
B_com_csv = 'B_com.csv'
C_com_csv = 'C_com.csv'
D_com_csv = 'D_com.csv'
E_com_csv = 'E_com.csv'


df_a = pd.read_csv(A_csv)
df_b = pd.read_csv(B_csv)
df_c = pd.read_csv(C_csv)
df_d = pd.read_csv(D_csv)
df_e = pd.read_csv(E_csv)

df_a_com = pd.read_csv(A_com_csv)
df_b_com = pd.read_csv(B_com_csv)
df_c_com = pd.read_csv(C_com_csv)
df_d_com = pd.read_csv(D_com_csv)
df_e_com = pd.read_csv(E_com_csv)

icons_c = ['c1.png','c2.png','c3.png','c4.png','c5.png']
icons_b = ['b1.png','b2.png','b3.png','b4.png','b5.png']

df_a_com = df_a_com.sort_values(by='current_time', ascending=True)
df_b_com = df_b_com.sort_values(by='current_time', ascending=True)
df_c_com = df_c_com.sort_values(by='current_time', ascending=True)
df_d_com = df_d_com.sort_values(by='current_time', ascending=True)
df_e_com = df_e_com.sort_values(by='current_time', ascending=True)
# com status 0 is urgent, 1 is busy for nrmal, 2 is idle, 3 is send back the result to the head, 4 is local process
#item of df_com is {'current_time':time_a-start_time,'CurrentNode':CurrentNode,'nextNode':nextNode,'communication_time':0,'com_status':com_status,'sourceNode':sourceNode}


def read_images(image_list):
    images = []
    for image in image_list:
        try:
            images.append(mpimg.imread(image))
        except FileNotFoundError:
            images.append(None)  # Handle missing files gracefully
    return images


def save_and_show_fig(fig, file_path):
    # 保存图形为 PNG 文件
    fig.savefig(file_path)  # 保存为 PNG 格式

    # 检查文件是否已保存，然后在系统中打开它
    if os.path.exists(file_path):
        # 使用系统默认的图片查看器打开
        os.system(f"start {file_path}")  # Windows系统
        # os.system(f"open {file_path}")  # macOS系统
        # os.system(f"xdg-open {file_path}")  # Linux系统
    else:
        print(f"文件 {file_path} 未找到！")



# # Load images
images_icons_c = read_images(icons_c)
images_icons_b = read_images(icons_b)
img = mpimg.imread('c1.png')


# 创建一个包含5个节点的完全连接图
#status 0 is urgent, 1 is busy for nrmal, 2 is idle
node_color = ['#D55E00','#56B4E9','#D3D3D3']  #[red,blue,gray] colorblind friendly.  #
task_color1 = ['#CC79A7','#009E73'] #0 is urgent, 1 is normal
edge_color=['#D55E00','#56B4E9','#6E7783','#F0E442'] #[red,blue,gray,yellow],连接边的颜色
edgecolors='#ababab',  # 节点边缘颜色


bar_color_1= '#D55E00'  #light red, colorblind friendly
bar_color_2= '#56B4E9' #light blue, colorblind friendly
bg_color='lightgray'

nodes = ['A', 'B', 'C', 'D', 'E']
node_size_size = [4000,2500,2500,4000,2500]




def update(frame):

    global node_size_size
    global nodes
    global fig

    print(frame)

    node_colors = []
    labels = []

    #clear the all previous axes in case the overlab of the graphs
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1]) #add a new axes to the fit the whole figure



    

    #infos for each node
    I = nx.Graph()
    for df, name in zip([df_a,df_b,df_c,df_d,df_e], ["A","B","C","D","E"]):
        filtered_df = df[df['current_time'] <= frame]
        if filtered_df.empty:
            node_colors.append(node_color[2])
            labels.append('Node ' +name+'\n' + 'TS:0' + "\nNTS:0" + "\nIndexOfPar:N")
            icon = None #empty image
            Task_type = "Idle"
            taskProcessNum = [0,0]
            index_partition = 0
        else:
            taskProcessNum = filtered_df.tail(1)['taskProcessNum'].values[0]
            index_partition = filtered_df.tail(1)['index_partition'].values[0]
            status = int(filtered_df.tail(1)['status'].values[0])
            taskProcessNum = eval(taskProcessNum)
            index_partition = int(index_partition)
            node_colors.append(node_color[status])
            if status == 0:
                icon = images_icons_c[(index_partition-1)] #partition index starts from 1
                Task_type = "TS"
            elif status == 1:
                icon = images_icons_b[(index_partition-1)]
                Task_type = "NTS"
            else:
                icon = None #empty image
                Task_type = "Idle"
            if index_partition == 0:
                sample_info = 'Node ' +name+'\n' + 'TS#:' +str(taskProcessNum[0]) + "\nNTS#:" + str(taskProcessNum[1]) + "\nPartition:\n" +Task_type
            else:
                sample_info = 'Node ' +name+'\n' + 'TS#:' +str(taskProcessNum[0]) + "\nNTS#:" + str(taskProcessNum[1]) + "\nPartition:\n" + Task_type + '-'+ str(index_partition)

            #icon = images_icons_c[1] #test sample, delete later
            labels.append(sample_info)
            
        I.add_node(name,image=icon,Task_type=Task_type,index_partition=index_partition,taskProcessNum = taskProcessNum)


    #edge_info for each edge
    activated_edges = [] #store the activated edges
    activated_edges_color = [] #store the activated edges color
    for df_com, name in zip([df_a_com,df_b_com,df_c_com,df_d_com,df_e_com], ["A","B","C","D","E"]): #{'current_time':time_a-start_time,'CurrentNode':CurrentNode,'nextNode':nextNode,'communication_time':0,'com_status':com_status,'sourceNode':sourceNode}
        #遍历所有的点，如果该点的某个边的时间小于当前时间，那么就将该边的颜色设置为红色
        # Step 1: 找到所有开始传输的数据项 且停止时间早于指定时间
        started_transmissions = df_com[(df_com['com_status'] != 2) & (df_com['current_time'] <= frame)]
        # Step 2: 找到所有停止传输的数据项 (com_status 为 2 且停止时间早于指定时间)
        stopped_transmissions = df_com[(df_com['com_status'] == 2) & (df_com['current_time'] <= frame)]
        if len(started_transmissions) > len(stopped_transmissions):
            # Step 3: 如果 started_transmissions 更长，计算差值
            diff = len(started_transmissions) - len(stopped_transmissions)
            # Step 4: 输出 started_transmissions 队尾的差值大小项数
            active_started_transmissions = started_transmissions.tail(diff)
        else:
            active_started_transmissions = pd.DataFrame() #如果没有数据，就返回一个空的DataFrame

        for row in active_started_transmissions.itertuples():
            CurrentNode = row.CurrentNode
            nextNode = row.nextNode
            com_status = row.com_status # com status 0 is urgent, 1 is busy for nrmal, 2 is idle, 3 is send back the result to the head, 4 is local process
            if com_status == 0:
                activated_edges.append((CurrentNode,nextNode))
                activated_edges_color.append(edge_color[0]) #edge_color [urgent,nonurget,gray],连接边的颜色
            elif com_status == 1:
                activated_edges.append((CurrentNode,nextNode))
                activated_edges_color.append(edge_color[1])
            # elif com_status == 3:
            #     activated_edges.append((CurrentNode,nextNode))
            #     activated_edges_color.append(edge_color[3])

    #Main graph G
    undirected_G = nx.complete_graph(nodes)
    G = nx.DiGraph(undirected_G) # 将无向图转换为有向图
    # 设置圆周上的固定位置，使 A 位于最上方
    angle_offset = - np.pi / 2  # 使A位于顶部
    angle = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False) - angle_offset
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angle)}

    #Labels graph
    H = nx.complete_graph(labels)
    r_factor = 1.15
    angle_offset_H = (-np.pi / 2)*1.3 # 使A位于顶部
    angle_H = np.linspace(0, 2 * np.pi, len(labels), endpoint=False) - angle_offset_H
    pos_H = {node: (np.cos(a)*r_factor, np.sin(a)*r_factor) for node, a in zip(labels, angle_H)}

    #Icon graph
    icon_nodes = I.nodes()
    r_factor = 1.05 #radius factor
    angle_offset_I = (- np.pi / 2)*0.8 # 使A位于顶部
    angle_I = np.linspace(0, 2 * np.pi, len(icon_nodes), endpoint=False) - angle_offset_I
    pos_I = {node: (np.cos(a)*r_factor, np.sin(a)*r_factor-0.07) for node, a in zip(icon_nodes, angle_I)}


    
    # 绘制图形 for the main graph and labels graph
    #nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_size_size, font_size=20, font_weight='bold', edge_color=edge_color,width=3,edgecolors=edgecolors,linewidths=2)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size_size,edgecolors=edgecolors,linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=20, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=20, edge_color=edge_color[2], width=3,node_size=node_size_size) #draw the idle edges, fully connected
    nx.draw_networkx_edges(G, pos, edgelist=activated_edges, arrowstyle='->', arrowsize=20, edge_color=activated_edges_color, width=4,node_size=node_size_size) #draw the active edges, so that the communication edge will be on the top of the idle edge
    #enable the labels graph, H is the labels graph
    alpha_label = 0 #transparency for the labels graph, 0 means disable
    #nx.draw(H, pos_H, with_labels=True, node_color='#FFFFFF', node_size=4500,edge_color='black',width=0,node_shape="s",edgecolors='black',linewidths=1,font_size=10,alpha=alpha_label)

    #plt.title("PA-MDI DEMO")
    plt.margins(x=0.5, y=0.12)
    node_legend0 = mlines.Line2D([], [], color=node_color[0], marker='o', linestyle='None',markersize=10, label='Node is Busy for \nTime-Sensitive(TS) Tasks')
    node_legend1 = mlines.Line2D([], [], color=node_color[1], marker='o', linestyle='None',markersize=10, label='Node is Busy for \nNon-Time-Sensitive(NTS) Tasks')
    node_legend2 = mlines.Line2D([], [], color=node_color[2], marker='o', linestyle='None',markersize=10, label='Node is Idle')
    node_legend3 = mlines.Line2D([], [], color=node_color[0], marker='>', linestyle='-',markersize=10, label='Transmission of \nTime-Sensitive(TS) Tasks')
    node_legend4 = mlines.Line2D([], [], color=node_color[1], marker='>', linestyle='-',markersize=10, label='Transmission of \nNon-Time-Sensitive(NTS) Tasks')
    node_legend5 = mlines.Line2D([], [], color=edge_color[2], marker='', linestyle='-',markersize=10, label='No Transmission')

    node_legend_text = mlines.Line2D([], [], color='black', marker='*', linestyle='None',markersize=0, label='TS#:The number of completed \nurgent partitions\nNTS#:The number of completed \nnon-urgent partitions\nPartition: The index of the \npartitioned task')
    legend1 = plt.legend(handles=[node_legend0,node_legend1,node_legend2,node_legend3,node_legend4,node_legend5], loc='upper right',fontsize=9.5)
    #legend2 = plt.legend(handles=[node_legend_text], loc='upper left',fontsize=11,handlelength=0, handletextpad=0)
    plt.gca().add_artist(legend1)
    plt.box(False) #remove the box of the plot

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
    icon_center = icon_size / 2.0
    pos_I_img = copy.deepcopy(pos_I)
    pos_I_bar = copy.deepcopy(pos_I)
    pos_I_note_source = copy.deepcopy(pos_I)

    #Add the NN icons and progress bars for each node
    for n in I.nodes:
        #Dwaw the icons graph
        #pos adjustment for the icons
        if n == "A":
            pos_I_img[n] = (pos_I_img[n][0]-0.75,pos_I_img[n][1]+0.16)
        if n == "B":
            pos_I_img[n] = (pos_I_img[n][0]-0.35,pos_I_img[n][1]+0.1)
        if n == "C":
            pos_I_img[n] = (pos_I_img[n][0]-0.1,pos_I_img[n][1]+0.1)
        if n == "D":
            pos_I_img[n] = (pos_I_img[n][0]+0.65,pos_I_img[n][1]+0.3)
        if n == "E":
            pos_I_img[n] = (pos_I_img[n][0],pos_I_img[n][1]+0.08) # D and E y position adjustment, up 
        xf, yf = tr_figure(pos_I_img[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        if I.nodes[n]["image"] is not None:
            ax_icon = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
            ax_icon.imshow(I.nodes[n]["image"])
            ax_icon.axis("off")
            ax_icon_text = fig.add_axes([xa-icon_size*0.5, ya-icon_size*1.4,icon_size,icon_size])
            ax_icon_text.axis('off')
            if I.nodes[n]["index_partition"] == 1:
                partition_info = '1st Partition'
            elif I.nodes[n]["index_partition"] == 2:
                partition_info = '2nd Partition'
            elif I.nodes[n]["index_partition"] == 3:
                partition_info = '3rd Partition'
            elif I.nodes[n]["index_partition"] == 4 :
                partition_info = '4th Partition'
            else:
                print('icon error'+str(I.nodes[n]["index_partition"]))
                partition_info = str(I.nodes[n]["index_partition"])+'th Partition'
            
            info_node = partition_info + '\nof '+I.nodes[n]["Task_type"]
            ax_icon_text.text(0.5,0.5,info_node, ha='center', va='center', fontsize=10, color='black')


        #add bars for the progress of the tasks each node is processing, pos_I[n]是相较于icons的位置
        if n == "A":
            pos_I_bar[n] = (pos_I_bar[n][0]-0.1,pos_I_bar[n][1]) 
        if n == "B":
            pos_I_bar[n] = (pos_I_bar[n][0]-0.9,pos_I_bar[n][1]-0.35)
        if n == "C":
            pos_I_bar[n] = (pos_I_bar[n][0]-0.5,pos_I_bar[n][1]-0.35)
        if n == "D":
            pos_I_bar[n] = (pos_I_bar[n][0]-0.52,pos_I_bar[n][1]+0.03)
        if n == "E":
            pos_I_bar[n] = (pos_I_bar[n][0]+0.1,pos_I_bar[n][1]+0.29)
        xf, yf = tr_figure(pos_I_bar[n])
        xa, ya = tr_axes((xf, yf))
        ax_bars_node = fig.add_axes([xa, ya, 0.14, 0.07]) #Set ax size of the bar
        bar_height = 0.2
        taskProcessNum = I.nodes[n]["taskProcessNum"]
        #bars UT
        ax_bars_node.barh(0, 1, color=bg_color, height=bar_height) #background
        rect = ax_bars_node.barh(0, taskProcessNum[0]/sample_num, color=bar_color_1, height=bar_height)
        #ax_bars_node.bar_label(rect, ['UT:25'], label_type='edge',padding=3,fontsize=12)
        bar_info_UT = 'TS Num:'+str(taskProcessNum[0])
        ax_bars_node.text(0.5, -0.01, bar_info_UT, ha='center', va='center', fontsize=10, color='black')
        #bars Non-UT
        NonUT_bar_v_pos_adjust = 0.28 #vertical position adjustment compared to UT bar
        ax_bars_node.barh(NonUT_bar_v_pos_adjust, 1, color=bg_color, height=bar_height) #background
        rect = ax_bars_node.barh(NonUT_bar_v_pos_adjust, taskProcessNum[1]/sample_num, color=bar_color_2, height=bar_height)
        bar_info_nonUT = 'NTS Num:'+str(taskProcessNum[1])
        ax_bars_node.text(0.5, NonUT_bar_v_pos_adjust-0.01, bar_info_nonUT, ha='center', va='center', fontsize=10, color='black')
        #ax_bars_node.bar_label(rect, ['Non-UT:100'], label_type='edge',padding=3,fontsize=12)

        ax_bars_node.spines['top'].set_visible(False)
        ax_bars_node.spines['bottom'].set_visible(False)
        ax_bars_node.spines['left'].set_visible(False)
        ax_bars_node.spines['right'].set_visible(False)
        ax_bars_node.set_xticks([])
        ax_bars_node.set_yticks([])

        #Add notes for the source node A and D
        if n == "A" or n == "D":
            if n == "A":
                pos_I_note_source[n] = (pos_I_note_source[n][0]-1.7,pos_I_note_source[n][1]-0.05)
                note_source = 'Node A has \nTime-Sensitive\n(TS) Data'
            if n == "D":
                pos_I_note_source[n] = (pos_I_note_source[n][0]+0.8,pos_I_note_source[n][1]+0.15)
                note_source = 'Node D has \nNon-Time-Sensitive\n(NTS) Data'
            xf, yf = tr_figure(pos_I_note_source[n])
            xa, ya = tr_axes((xf, yf))
            ax_source_note = fig.add_axes([xa, ya, 0.2, 0.12]) #Set ax size of the bar
            ax_source_note.axis('off')
            ax_source_note.text(0.5,0.5,note_source, ha='center', va='center', fontsize=13, color='black',fontweight='bold')

    #Add the comunication edge between the nodes
    #print(G.edges())
    #print(G['A']['B']) #get the edge data of A-B
    # G['A']['B']['color'] = 'red'
    # print(G['A']['B'])
    # print(G['B']['A'])



        
        
        










#fig, ax = plt.subplots(0,figsize=(10,6),dpi=300)
fig = plt.figure(figsize=(10,6),dpi=300) #initiate the figure, there is no ax at all
print(fig.get_axes())

# # 创建动画
anim = FuncAnimation(fig, update, frames=np.arange(0, 300, 0.1), interval=1)
FFwriter = animation.FFMpegWriter(fps=50, extra_args=['-vcodec', 'libx264'])
anim.save('MDI-network-2-topo.mp4', writer=FFwriter)

# update(100)
# save_and_show_fig(fig, 'temp.png')

