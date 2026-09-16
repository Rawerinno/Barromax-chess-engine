import chess
from engine import find_best_move


def escolher_cor():
    escolha = input("Queres jogar de brancas ou pretas? Escreve b ou p: ").strip().lower()

    if escolha == "p":
        return chess.BLACK

    return chess.WHITE


def mostrar_tabuleiro(board):
    print()
    print(board.unicode(borders=True))
    print()


def pedir_jogada(board):
    while True:
        texto = input("A tua jogada: ").strip().lower()

        if texto == "sair":
            return None

        if texto == "fen":
            print(board.fen())
            continue

        if texto == "ajuda":
            print("Escreve jogadas em formato UCI.")
            print("Exemplos: e2e4, g1f3, e7e8q")
            print("Comandos: sair, fen, ajuda")
            continue

        try:
            move = chess.Move.from_uci(texto)
        except ValueError:
            print("Formato inválido. Exemplo correto: e2e4")
            continue

        if move not in board.legal_moves:
            print("Jogada ilegal.")
            continue

        return move


def jogar():
    board = chess.Board()
    human_color = escolher_cor()

    depth = 3

    print()
    print("Jogo iniciado.")
    print("Escreve movimentos assim: e2e4, g1f3, e7e8q")
    print("Para sair, escreve: sair")
    print("Para ver a posição FEN, escreve: fen")
    print()

    while not board.is_game_over(claim_draw=True):
        mostrar_tabuleiro(board)

        if board.turn == human_color:
            move = pedir_jogada(board)

            if move is None:
                print("Jogo terminado.")
                return

            board.push(move)

        else:
            print("Engine a pensar...")

            move, score = find_best_move(board, depth=depth)

            if move is None:
                break

            print(f"Engine joga: {move}")
            print(f"Avaliação: {score}")

            board.push(move)

    mostrar_tabuleiro(board)

    outcome = board.outcome(claim_draw=True)

    print("Fim do jogo.")
    print("Resultado:", board.result(claim_draw=True))

    if outcome is not None:
        if outcome.winner == chess.WHITE:
            print("Vencedor: brancas")
        elif outcome.winner == chess.BLACK:
            print("Vencedor: pretas")
        else:
            print("Empate")


if __name__ == "__main__":
    jogar()