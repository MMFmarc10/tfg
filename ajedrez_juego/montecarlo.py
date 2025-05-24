

import chess
import math

import numpy as np
import chess

import torch

from torchvision.transforms import ToTensor



capa_pieza = {'p': 1, 'r': 3, 'n': 5, 'b': 7, 'q': 9, 'k': 11, 'P': 0, 'R': 2, 'N': 4, 'B': 6, 'Q': 8, 'K': 10}

numero_letra = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

letra_numero = {'a': 0, 'b': 1, 'c': 2, 'd':3 , 'e': 4, 'f': 5, 'g':6 , 'h': 7}



def representacion_tablero(board,verbose=False):
    fen = board.fen().split()
    tablero_piezas_fen = fen[0]
    turno_fen = fen[1]
    enroque_fen = fen[2]
    enpassant_fen = fen[3]

    tablero_r = np.zeros((8,8,22),dtype=np.float32)

    fila=0
    columna=0
    
    for caracter in tablero_piezas_fen:
        
        if not caracter.isdigit():
         
            if caracter == '/':
                fila+=1
                columna=0
           
            else:
    
                tablero_r[fila][columna][capa_pieza[caracter]]=1
                columna+=1
                
        else:
            columna+= int(caracter)

    if turno_fen == 'w':
        tablero_r[:, :, 12] = 1
    if 'K' in enroque_fen:
        tablero_r[:, :, 13] = 1
    if 'k' in enroque_fen:
        tablero_r[:, :, 14] = 1
    if 'Q' in enroque_fen:
        tablero_r[:, :, 15] = 1
    if 'q' in enroque_fen:
        tablero_r[:, :, 16] = 1
           

    if enpassant_fen != "-":
        tablero_r[8-int(enpassant_fen[1])][letra_numero[enpassant_fen[0]]][17] = 1


    for square in chess.SQUARES:
        row = 7 - (square // 8)
        col = square % 8  
        tablero_r[row, col, 18] = len(board.attackers(chess.WHITE, square))/10
        tablero_r[row, col, 19] = len(board.attackers(chess.BLACK, square))/10

    for move in board.legal_moves:
        to_square = move.to_square
        row = 7 - (to_square // 8)
        col = to_square % 8
        tablero_r[row][col][20] = 1

    if board.is_check():
        tablero_r[:, :, 21] = 1

    

    if turno_fen == 'b':
        tablero_r = np.flip(tablero_r[:, :, :22], axis=(0, 1)).copy()

    if verbose:       
        for capa in range(22):
             for fila in range(8):
                 for columna in range(8):
                     print(tablero_r[fila][columna][capa], end="")
                     print(" ",end="")
                 print(" ")
             print("\n")



    return tablero_r


    
def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()
    
def obtener_distribucion(board,model,device):
    model.eval()
    input = representacion_tablero(board)

    transform = ToTensor()
    tablero_tensor = transform(input).unsqueeze(0)

    with torch.no_grad():  # Asegúrate de que no se realice el cálculo del gradiente
        tablero_tensor= tablero_tensor.to(device)
        outputs,eval = model(tablero_tensor)  # Realiza la predicción
    outputs = outputs.squeeze(0).detach().cpu().numpy()
    np.set_printoptions(suppress=True, precision=2)
    #print(outputs)
    value= eval.cpu().item()

  
  

    movimientos_legales= [move.uci() for move in board.legal_moves]

    
    valores_mapa_activacion = []


    for uci_move in movimientos_legales:

        move= chess.Move.from_uci(uci_move)
        board.push(move)
        if board.is_checkmate():
            board.pop()  # Deshacer el movimiento para no modificar el estado del tablero
            valores_mapa_activacion.append(1)
            return  valores_mapa_activacion,[uci_move],1.0
        board.pop()  # Deshacer el movimiento para continuar buscando
    
              
        if board.turn == chess.WHITE:

            columna_inicial = letra_numero[uci_move[0]]
            fila_inicial = 8-int(uci_move[1])

            columna_final = letra_numero[uci_move[2]]
            fila_final = 8-int(uci_move[3])
                
        else:

            columna_inicial = 7-letra_numero[uci_move[0]]
            fila_inicial = 7-(8-int(uci_move[1]))

            columna_final = 7-letra_numero[uci_move[2]]
            fila_final = 7-(8-int(uci_move[3]))

        valores_mapa_activacion.append(outputs[0][fila_inicial][columna_inicial]+outputs[1][fila_final][columna_final])

    #valores_mapa_activacion /= np.sum(valores_mapa_activacion)
  

    valores_mapa_activacion = softmax(valores_mapa_activacion)

   # print(valores_mapa_activacion)
    return  valores_mapa_activacion,movimientos_legales,value




class Node:
    def __init__(self,prior,board,move=None):
        self.prior = prior
        self.visit_count= 0
        self.value_sum = 0
        self.children=[]
        self.board= board
        self.move = move

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children)>0

    def expand(self,list_distribution,moves):
        
        for i,uci_move in enumerate(moves):

            
            move = chess.Move.from_uci(uci_move)
            board_child = self.board.copy()
            board_child.push(move)
            
            node = Node(list_distribution[i],board_child,uci_move)
            self.children.append(node)
            #node.print_values()
            
    def print_values(self):
        print(self.move)
        print(self.prior)
        print(self.visit_count)
        print(self.board)
        

    def select(self):

        mejor_ucb = -np.inf
        mejor_nodo= None
        for node in self.children:
            ucb_score = self.get_ucb_score(node)
            if ucb_score>mejor_ucb:
                mejor_ucb = ucb_score
                mejor_nodo = node

       # mejor_nodo.print_values()
        return mejor_nodo
        

    def get_ucb_score(self,child):
        prior_score = child.prior *2 * math.sqrt(self.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            # The value of the child is from the perspective of the opposing player
            value_score = -child.value()
        else:
            value_score = 0

        return value_score + prior_score
        


class Chess_MCTS:
    def __init__(self ,num_simulations,modelo,device):

        self.simulations = num_simulations
        self.modelo = modelo
        self.device = device
        
    def iniciar(self,inicial_board):

        root = Node(1,inicial_board)
        
        for _ in range(self.simulations):

            node = root
            search_path = [node]

            #selection
            while node.is_expanded():
                node = node.select()
                search_path.append(node)

            terminada,value = self.partida_terminada(node.board)
            if not terminada:
                distribucion,movimientos,value = obtener_distribucion(node.board,self.modelo,self.device)
                node.expand(distribucion,movimientos)
            
            self.backpropagate(search_path, value)
        moves=[]
        distribution=[]
        for i,child in enumerate(root.children):
            moves.append(child.move)
            distribution.append(child.visit_count)
            #child.print_values()
        return moves,distribution,root.value()
    
    
    def partida_terminada(self,board):
        # Verificar si hay jaque mate
        if board.is_checkmate():
           
            return True,-1

        # Verificar si hay ahogado (stalemate)
        if board.is_stalemate():
            return True,0

        # Verificar si hay tablas por insuficiencia de material
        if board.is_insufficient_material():
            return True,0

        # Verificar regla de los 50 movimientos sin captura ni avance de peón
        if board.can_claim_fifty_moves():
            return True,0

        # Verificar regla de tres repeticiones
        if board.is_repetition(3):
            return True,0

        # Si ninguna condición de finalización se cumple, la partida no ha terminado
        return False,0


    def backpropagate(self, search_path, value):
        oponente = 1
        for node in reversed(search_path):
            node.value_sum += value*oponente
            node.visit_count += 1
            oponente = oponente*-1
        
class BotMonteCarlo:
    def __init__(self,simulaciones,modelo,device):

        self.simulaciones = simulaciones
        self.modelo = modelo
        self.device= device
        self.mcts = Chess_MCTS(self.simulaciones,self.modelo,self.device)

     
    def obtener_movimiento(self,board):

        moves,distribution,value=self.mcts.iniciar(board)
        print(moves)
        print(distribution)
        print(value)
        index_max=distribution.index(max(distribution))
        move = chess.Move.from_uci(moves[index_max])

        return move,value