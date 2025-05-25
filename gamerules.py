import gamerules

N_ROWS = 6
N_COLS = 7

import numpy as np
import matplotlib.pyplot as plt


from abc import ABC, abstractmethod
import base64,json, select, sys, os

import subprocess


class ProcessingTimeout(Exception):
    """ Marker for timeout """
    pass

class PongFailure(Exception):
    """ failure respond in ping command """
    pass

# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Board:
    def __init__(self):
        self.board = np.zeros((N_ROWS, N_COLS))  # 0- empty, 1 - 1st Player (starts 1st), -1 - 2nd Player
        # 0 - empty, >0 - components of 1st Player, <0 - components of 2nd Player
        self.components = np.zeros((N_ROWS, N_COLS))
        self.components4 = np.zeros((N_ROWS, N_COLS))
        self.componentID = 0
        self.component4ID = 0

    def resetBoard(self):
        self.board = np.zeros((N_ROWS, N_COLS))  # 0- empty, 1 - 1st Player (starts 1st), -1 - 2nd Player
        # 0 - empty, >0 - components of 1st Player, <0 - components of 2nd Player
        self.components = np.zeros((N_ROWS, N_COLS))
        self.components4 = np.zeros((N_ROWS, N_COLS))
        self.componentID = 0
        self.component4ID = 0

    def showBoard(self):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        plt.sca(ax1)
        plt.matshow(self.board, fignum=False)
        plt.colorbar()
        plt.title("Board")
        plt.sca(ax2)
        plt.matshow(self.components, fignum=False)
        plt.colorbar()
        plt.title("Components (diag connections)")
        plt.sca(ax3)
        plt.matshow(self.components4, fignum=False)
        plt.colorbar()
        plt.title("Components (hor&vert connections)")
        plt.show()

    def plotBoard(self, player1, player2):
        f, ax1 = plt.subplots(1, 1, figsize=(7, 7))
        for i in range(N_ROWS + 1):
            plt.plot([0, N_COLS], [i, i], 'k-', linewidth=3)
        for j in range(N_COLS + 1):
            plt.plot([j, j], [0, N_ROWS], 'k-', linewidth=3)
        indicesFirst = np.where(self.board == 1)
        plt.plot(indicesFirst[1] + 0.5, N_ROWS - indicesFirst[0] - 0.5, 'o', color=(0.77, 0.35, 0.067), markersize=18, label = player1.getName())
        indicesFirst = np.where(self.board == -1)
        plt.plot(indicesFirst[1] + 0.5, N_ROWS - indicesFirst[0] - 0.5, 'o', color=(0.77, 0.77, 0.77), markersize=18, label = player2.getName())
        plt.axis('equal')
        plt.axis("off")
        plt.legend()
        plt.show()

    def updateBoard(self, action, value):
        row = np.max(np.where(self.board[:, action] == 0))
        self.board[row, action] = value
        self.updateComponents(row, action, value)
        self.updateComponents4(row, action, value)
        return 1

    def getPossibleActions(self):
        return np.unique(np.where(self.board == 0)[1])

    # update components regarding diagonal connections
    def updateComponents(self, row, column, value):
        rowFrom = np.max([0, row - 1])
        rowTo = np.min([row + 2, N_ROWS])
        colFrom = np.max([0, column - 1])
        colTo = np.min([column + 2, N_COLS])
        componentSub = self.components[rowFrom:rowTo, colFrom:colTo]
        IS = np.sign(componentSub) == np.sign(value)
        uniqueComponents = np.unique(componentSub[IS])
        if len(uniqueComponents) == 0:
            self.componentID += 1
            minComponent = self.componentID * value
        else:
            minComponent = np.min(uniqueComponents)
        IS = np.isin(self.components, uniqueComponents)
        self.components[IS] = minComponent
        self.components[row, column] = minComponent

    # update components regarding horizontal and vertical connections
    def updateComponents4(self, row, column, value):
        uniqueComponents = list()
        if row > 0:
            if np.sign(self.board[row - 1, column]) == np.sign(value):
                uniqueComponents.append(self.components4[row - 1, column])
        if row < (N_ROWS - 1):
            if np.sign(self.board[row + 1, column]) == np.sign(value):
                uniqueComponents.append(self.components4[row + 1, column])
        if column > 0:
            if np.sign(self.board[row, column - 1]) == np.sign(value):
                uniqueComponents.append(self.components4[row, column - 1])
        if column < (N_COLS - 1):
            if np.sign(self.board[row, column + 1]) == np.sign(value):
                uniqueComponents.append(self.components4[row, column + 1])
        if len(uniqueComponents) == 0:
            self.component4ID += 1
            minComponent = self.component4ID * value
        else:
            minComponent = np.min(uniqueComponents)
        IS = np.isin(self.components4, uniqueComponents)
        self.components4[IS] = minComponent
        self.components4[row, column] = minComponent

    # check if components form closed boundary and consists other value inside (victory)
    def checkVictory(self, column, value):
        indices = np.where(self.board[:, column] == 0)[0]
        if len(indices) > 0:
            row = (np.max(indices) + 1)
        else:
            row = 0
        rowFrom = np.max([0, row - 1])
        rowTo = np.min([row + 2, N_ROWS])
        colFrom = np.max([0, column - 1])
        colTo = np.min([column + 2, N_COLS])
        componentSub = self.components[rowFrom:rowTo, colFrom:colTo]
        IS = np.sign(componentSub) == np.sign(value)
        uniqueComponents = np.unique(componentSub[IS])
        for comp in uniqueComponents:
            if np.count_nonzero(componentSub == comp) > 2:
                cont = self.checkOtherValueInside(comp, row, column, value)
                if cont:
                    return True

        return False

    # check if the closed boundary has a value inside it
    def checkOtherValueInside(self, comp, row, column, value):
        foundOtherValueInside = False
        componentsToCheck = list()
        if row > 0:
            if np.sign(self.board[row - 1, column]) != np.sign(value):
                componentsToCheck.append(self.components4[row - 1, column])
        if row < (N_ROWS - 1):
            if np.sign(self.board[row + 1, column]) != np.sign(value):
                componentsToCheck.append(self.components4[row + 1, column])
        if column > 0:
            if np.sign(self.board[row, column - 1]) != np.sign(value):
                componentsToCheck.append(self.components4[row, column - 1])
        if column < (N_COLS - 1):
            if np.sign(self.board[row, column + 1]) != np.sign(value):
                componentsToCheck.append(self.components4[row, column + 1])
        uniqueComponents = np.unique(componentsToCheck)
        for component in uniqueComponents:
            allClosedForComponent = True
            indices = np.where(self.components4 == component)

            for i in range(len(indices[0])):
                checkRow = indices[0][i]
                checkColumn = indices[1][i]
                # if is open to bordering line
                if (checkRow == 0) or (checkRow == (N_ROWS - 1)) or \
                        (checkColumn == 0) or (checkColumn == (N_COLS - 1)):
                    allClosedForComponent = False
                    break
                possibleComps = np.array(
                    [comp, self.components[checkRow, checkColumn]])  # possible either the closing or same
                if (self.components[checkRow - 1, checkColumn] not in possibleComps) or \
                        (self.components[checkRow + 1, checkColumn] not in possibleComps) or \
                        (self.components[checkRow, checkColumn - 1] not in possibleComps) or \
                        (self.components[checkRow, checkColumn + 1] not in possibleComps):
                    allClosedForComponent = False
                    break
            if (allClosedForComponent):
                #print("Closed indices:")
                #print(indices)
                foundOtherValueInside = True
                break
        return foundOtherValueInside

    # prepare board to have it from the player's perspective
    def prepareBoardForPlayer(self, value):
        # always 1 for the player, 0 for empty values, -1 for opponent
        playerBoard = np.zeros_like(self.board)
        IS = self.board == value
        playerBoard[IS] = 1
        IS = self.board == -1 * value
        playerBoard[IS] = -1
        return playerBoard

    def sampleGame(self, player1, player2):
        """ play sample game
            returns -1 - player1 won
            returns 0 - draw
            returns 1 - player2
        """
        assert player1 != player2

        players = [{'p': player1, 'w': player1, 'l': player2, 'n': 1}, {'p': player2, 'w': player2, 'l': player1,
                                                                        'n': -1}]  # p - current player, 'w' player which won on this tep, 'l' - player on lose case (for example crash, time and etc)
        startValue = {player1: 1, player2: -1}  # player p1 starts first

        player1.getName(alt_timeout=10)
        player2.getName(alt_timeout=10)

        gameFinished = False
        while (not gameFinished):
            for player in players:

                try:
                    action = player['p'].getAction(copy.deepcopy(self), startValue[player['p']])
                except ProcessingTimeout as e:
                    print("Player `%s' lost due timeout, winner `%s'" % (player['p'].getName(), player['l'].getName()))
                    gameFinished = True
                    return player['n']
                    break

                except Exception as e:
                    print("Player `%s' lost due crash, winner `%s'" % (player['p'].getName(), player['l'].getName()))
                    print(e)
                    gameFinished = True
                    return player['n']
                    break

                print("Player " + player['p'].getName() + " : " + str(action))
                self.updateBoard(action, startValue[player['p']])

                gameFinished = self.checkVictory(action, startValue[player['p']])
                if gameFinished:
                    print(player['p'].getName() + " won")
                    return player['n']
                    break
                if len(self.getPossibleActions()) == 0:
                    print("No possible actions left, draw")
                    gameFinished = True
                    return 0;
                    break



class Player(ABC):

  name : str # player name


  def __init__(self, name):
    self.name = name

  def getName(self):
    return self.name

  @abstractmethod
  def getAction(self, board, startValue):
    """ All logic going here """
    possibleActions = self.getPossibleActions(board.board)
    return np.random.choice(possibleActions)

  @abstractmethod
  def newGame(new_opponent : bool):
    pass


  def getPossibleActions(self, board):
    return np.unique(np.where(board == 0)[1])


  def stop_playing(self):
    """ informs that game ended """
    self.runing = False
    return None

  def ping(self):
    return "pong"

  def serve_via_std(self):
      """ serve remote player commands via stdin and stdout """

      running = True

      while running:
        l = sys.stdin.readline(524288); # 512 kb board upto 370x370 can be supported
        l = l.strip() # strip trailing '\n'
        
        packet = json.loads(base64.b85decode(l))

        method = packet[0]
        parameters = packet[1]
        if method == "getAction":
          parameters = [parameters[0], parameters[1]]


        m = getattr(self, method)
        response = m(*parameters)


        sys.stdout.write("%s\n" % (base64.b85encode(json.dumps({'res' : response }, cls=NpEncoder).encode()).decode(),));        
        sys.stdout.flush()

 



class RemotePlayerServer(Player):
  """ Represents remote player server part
  """

  def __init__(self, remoteplayer_program, secure_mode = False, default_timeout = 5):
        """
         :param secure_mode: if true run program via docker image "game-image"
        """
        super().__init__("Remote player " + remoteplayer_program)

        self.default_timeout = default_timeout

        self.cached_name = None
        self.p = None

        my_env = os.environ.copy()
        my_env["PLAYER1_REMOTE"] = "yes"

        if secure_mode:
          if sys.platform == 'linux':
            import shlex
            program_dir = os.path.abspath(os.path.dirname(remoteplayer_program))
            program = os.path.basename(remoteplayer_program)
            cmd = f"docker run -e PLAYER1_REMOTE -m 16g --cpus=16 --network none -v {shlex.quote(program_dir)}:/game:ro --rm -i game-image:latest python {shlex.quote(program)}"
            cmd_popen = shlex.split(cmd)

          else:
            raise NotImplementedError("docker image implemented only in linux")
        else:
         cmd_popen = [sys.executable,remoteplayer_program]

        self.cmd_popen = cmd_popen
        self.my_env = my_env

        self._start_process()


  def _start_process(self):
       # pipe has extra size, set text_mode (read call always reads whole line)
        self.p = subprocess.Popen(self.cmd_popen, env=self.my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,stdin =subprocess.PIPE, bufsize = 1024*1024, text=True)
  
  def __del__(self):
      if self.p:
        self.p.kill()
        self.p.wait()

  def communicate(self,method,params,alt_timeout = None):
      data = base64.b85encode(json.dumps([method,params]).encode()).decode()

      self.p.stdin.write("%s\n" %(data,));
      self.p.stdin.flush();

      if select.select([self.p.stdout.fileno()], [], [], self.default_timeout if alt_timeout is None else alt_timeout)[0]:
           out = os.read(self.p.stdout.fileno(), 524288).decode().strip()
      else:
           raise ProcessingTimeout();
           
      result = json.loads(base64.b85decode(out))['res']
      return result


  def getName(self,alt_timeout = None) -> str :
        if not self.cached_name:
          self.cached_name = self.communicate('getName',[],alt_timeout = alt_timeout)
        return self.name + ":" + self.cached_name

  def getAction(self, board : gamerules.Board, startValue) -> float :
        return self.communicate('getAction', [board, startValue])

  def  newGame(self, new_opponent : bool):
        self.communicate('newGame',[new_opponent])

  def  ping(self,timeout):
       r = self.communicate('ping',[],alt_timeout=timeout)
       if r != "pong":
         raise PongFailure()

  def kill_process(self):
    self.p.kill();
    self.p.wait();

  def stop_playing(self):
        try:
          return self.communicate('stop_playing',[])
        except Exception as e: # ignore exceptions
          pass

  def is_alive(self):
    """ check if process is alive """
    return self.p.poll() == None

  def check_for_restart(self):
    if not self.is_alive():
      self._start_process()

