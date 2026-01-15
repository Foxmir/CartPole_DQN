# src/utils/wandb_login.py

import wandb
import os
import json
from datetime import datetime, timedelta

from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__)

WANDB_PROJECT = "RL_Project_Data" # 常量-有效的项目名！
WANDB_ENTITY = "foxmir-stanford-university" # 常量-有效的用户实体名（用户名+公司）
WANDB_JOB_TYPE = "training" # 常量-默认的标签名称

def load_secret_key(secret_file="secrets.json") -> str | None: # 尝试加载包含密钥的json文件，这个模块功能虽然是加载文件+加载api，但如果文档打开成功，api加载失败并不会中断，只会返回none，然后抛给新函数（我们这里是函数嵌套，所以是内层函数抛给外层函数），并验证这个api登录是否成功，然后提供报错信息
    logger.info(f"尝试打开wandb的json配置文件并读取api...")
    api_key = None # 如果try成功就正常赋值，并传递到login_wandb中；如果失败，就还是none，传递到login_wandb中，又这个新函数来告知我们api获取失败。
    try:
        project_root  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 当前正在执行文件的绝对路径向上追溯3级别，到根目录
        secret_path = os.path.join(project_root, secret_file)
        logger.info(f"已经读取到wandb的json配置文件的完整路径: '{secret_path}'")
        with open(secret_path, 'r', encoding='utf-8') as f:
            secrets = json.load(f) # 这里读取的是 键+值
            api_key = secrets.get("WANDB_API_KEY") # 这里是用的(字典.获取键值)的get方法。这个常量名就是键名
            if api_key: # 成功
                logger.info("api密钥获取成功!")
            else: # 失败
                logger.warning(f"!!!在密钥路径'{secret_path}'下，名为 'WANDB_API_KEY' 的api密钥获取失败!!!") # 我们不再这里raise终端程序，而是让主函数在主逻辑报告
    except FileNotFoundError:
        logger.error(f"配置文件未找到: '{secret_path}'")
    except json.JSONDecodeError as e:
        logger.error(f"配置文件 '{secret_path}' 格式无效或解析错误: {e}", exc_info=True) 
    except Exception as e:
        logger.error(f"加载配置文件 '{secret_path}'时发生非预设的错误：{e}",exc_info=True)
    return api_key # 如果赋值成功正常赋值，如果失败，依然是None
    
def login_wandb() -> bool: # # 尝试使用本地密钥文件登录W&B
    logger.info(f"开始尝试登录 W&B ...")
    api_key = load_secret_key() # 调用上面的函数，实现模块内的封装，保证功能完整，外部仅需调用一个函数，相对简洁
    try:
        wandb.login(key=api_key, relogin=True)
        logger.info("W&B 登录成功! ")
        return True # 这将是主函数中报告的判断条件
    except Exception as e:
        logger.error(f"登录无法完成: {e}",exc_info=True) # 配合内部load_secret的报错，上下两条信息，足以定位问题
        return False # 这将是主函数中报告的判断条件

def create_wandb_run(run_name: str,config):  # wb依靠run对象记录消息
    try:
        run = wandb.init(
            project=WANDB_PROJECT,       # 使用在本模块定义的常量
            entity=WANDB_ENTITY,         # 使用在本模块定义的常量
            config=config,               # 传递完整的实验配置
            name=run_name,                # 使用传入或生成的运行名称，通常是从默认配置文件中传入
            job_type=WANDB_JOB_TYPE,      # 标签区分训练或者分析，方便快速筛选（糟糕的是，这个接口没有对外暴漏，会一直显示当前代码开头的默认常量training，懒得改了）
            settings=wandb.Settings(x_disable_stats=True) # 停掉CPU/GPU/内存等系统指标的采集,以减少开销
        )
        logger.info(f"W&B 名为'{run.name}'的 run 对象创建成功！请访问: {run.url} ")
        logger.info(f" (已经记录到项目: '{WANDB_PROJECT}', 账号实体: '{WANDB_ENTITY}')")
    except Exception as e:
        logger.error(f"创建 W&B 的run对象失败, 发生错误：'{e}'",exc_info=True)
        raise
    
    try:
        current_time = (datetime.now() + timedelta(hours=8)).strftime("%Y%m%d-%H%M") # 专业就从UTC改为了UTC+8的背景时间
        run.name = f"{wandb.config['agent']['name']}-{current_time}" # 例如 "智能体名字-20231027-1030",此处是从被sweep对象中拿取的名字再次做处理，便于区分比如普通运行，bayes运行，grid运行。wandb.config[agent.name]尝试但报错了，似乎必须严格按照嵌套结构提取，特别留意，双引号内部必须单引号
        logger.info(f"W&B运行名称修改成功: {run.name}")
        return run
    except Exception as e:
        logger.error(f"修改run对象名称失败, 发生错误：'{e}'",exc_info=True)
        raise