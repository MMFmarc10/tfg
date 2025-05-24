import tkinter as tk
import chess
from PIL import Image, ImageTk

import chess


import torch


import resnet
import montecarlo



class ChessBoard(tk.Canvas):
    def __init__(self, parent, gameboard,bot, turn,size=600, rows=8, cols=8):
        super().__init__(parent, width=size, height=size)
        self.size = size
        self.rows = rows
        self.cols = cols
        self.bot=bot
        self.square_size = size // rows
        self.pieces = {}
        self.selected_piece = None
        self.board = gameboard
        self.casilla_origen = None
        self.casilla_final = None
        self.images = self.load_images()
        self.paint_board()
        self.bind("<Button-1>", self.on_click)
        self.pintar_piezas()
        if turn == 1:
            self.bot_move()


    def load_images(self):
        images = {}
        pieces = ['p', 'r', 'n', 'b', 'q', 'k']
        colors = ['white', 'black']
        for color in colors:
            for piece in pieces:
                image_name = f"{color}_{piece}.png"
                image_path = f"imagenes/{image_name}"
                image = Image.open(image_path)
                image = image.resize((self.square_size, self.square_size), Image.Resampling.LANCZOS)
                images[image_name] = ImageTk.PhotoImage(image)
           
        return images

    def paint_board(self):
        colors = ["#DDB88C", "#A66D4F"]
        for row in range(self.rows):
            for col in range(self.cols):
                color = colors[(row + col) % 2]
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.create_rectangle(x1, y1, x2, y2, fill=color, tags="square", outline="")

    def pintar_piezas(self):
        piece_symbols = {
            'r': 'black_r.png', 'n': 'black_n.png', 'b': 'black_b.png', 'q': 'black_q.png',
            'k': 'black_k.png', 'p': 'black_p.png', 'R': 'white_r.png', 'N': 'white_n.png',
            'B': 'white_b.png', 'Q': 'white_q.png', 'K': 'white_k.png', 'P': 'white_p.png'
        }
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                row, col = divmod(square, 8)
                self.add_piece(piece_symbols[piece_symbol], (7 - row, col), piece_symbol)

    def add_piece(self, piece, position, piece_symbol):
        x, y = position
        x1 = y * self.square_size
        y1 = x * self.square_size
        piece_id = self.create_image(x1, y1, anchor='nw', image=self.images[piece], tags="piece")
        self.pieces[(x, y)] = (piece, piece_id)

    def on_click(self, event):
        col = event.x // self.square_size
        row = event.y // self.square_size
        if self.selected_piece:
            self.move_piece(self.selected_piece, (row, col))
           
            self.selected_piece = None
        else:
            self.selected_piece = (row, col)
            color = "#FF0000"  # Red color
            self.paint_square(row,col,color)

    def paint_square(self, row, col,color):
       
        x1 = col * self.square_size
        y1 = row * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        # Create a rectangle with only a red border
        self.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="highlight")

    def move_piece(self, from_pos, to_pos):
        # Convert (row, col) to chess square (0-63)
        from_square = chess.square(from_pos[1], 7 - from_pos[0])
        to_square = chess.square(to_pos[1], 7 - to_pos[0])
        
        piece = self.board.piece_at(from_square)
      
        if piece and piece.piece_type == chess.PAWN and (chess.square_rank(to_square) in [0, 7]):
     
            uci_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        else:
            uci_move = chess.Move(from_square, to_square)
        
        # Check if the move is legal
        if uci_move in self.board.legal_moves:
            # Push the move to the board

            
            self.board.push(uci_move)
            self.casilla_origen = from_pos
            self.casilla_final = to_pos

            if self.board.is_game_over():
                      
                resultado = self.board.result()
                if resultado == "1-0":  # Blancas ganan
                    print("Ganan Blancas!")
                elif resultado == "0-1":  # Negras ganan
                    print("Ganan Negras!")
                else:
                    print("Tablas!")

            self.update_board_ui()
            self.bot_move()
            print("---------------------------------------------------------------")
        else:
            print("Illegal move")
            self.update_board_ui()



    def bot_move(self):

        root.config(cursor="watch")
        root.update()  # Forzar actualización del cursor

        move,_= self.bot.obtener_movimiento(self.board)
        self.board.push(move)

          # Convertir las posiciones de las casillas al formato (row, col)
        self.casilla_origen = divmod(move.from_square, 8)
        self.casilla_final = divmod(move.to_square, 8)

        root.config(cursor="")
        root.update()

        if self.board.is_game_over():
                      
            resultado = self.board.result()
            if resultado == "1-0":  # Blancas ganan
                print("Ganan Blancas!")
            elif resultado == "0-1":  # Negras ganan
                print("Ganan Negras!")
            else:
                print("Tablas!")
        self.update_board_ui()


      
    def update_board_ui(self):
        self.delete("all")
        # Update the board UI
       
        self.paint_board()
        self.pintar_piezas()
        self.update_idletasks()



modelo = resnet.ResNet(12,256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Cargar los pesos guardados
  
modelo.load_state_dict(torch.load("modelo_pesos9.pth",map_location=torch.device("cpu")))
modelo = modelo.to(device)

bot =montecarlo.BotMonteCarlo(200,modelo,device)

root = tk.Tk()
gameboard = chess.Board()

board = ChessBoard(root, gameboard,bot,0)
board.pack()
root.mainloop()