-- 
-- Structure de la table 'personaldata'
--
DROP table if EXISTS 'PersonalData';
CREATE table `personaldata` (
    `id` int not NULL AUTO_INCREMENT DEFAULT,
    `nom` varchar(100) not NULL,
    `prenom` varchar(100) not null,
    `cptebank` int not null,
    `email` varchar(100) not null UNIQUE,
    `datempreunt` date,
    PRIMARY KEY(`id`),
)

--
-- Ajout des données dans la table 'personaldata'

LOCK TABLES 'personaldata' WRITE;
INSERT INTO 'personaldata' VALUES(1, 'lise', 'romuald' 12080, 'lise@gmail.com'), (2 'rodrigue', 'sensei' 20000, 'sensei@yahoo.fr'), (3, 'zoabar', 'raoul', 2390, 'raoul@outloook.com'), (4, 'remi', 'assogba' 2314, 'assogba@gmail.com'), (5, 'beryl','comlan',234654, 'marieelise@gmail.com'), (6, 'dedee', 'koumondji', 213426, 'dedekoumondji@gmail.com', 2023-03-12), (7, 'daniel','aballo', 231324n, 'amiraniel@gmail.com', 2022-12-23), (8, 'zaniel','dansou', 2500, 'zanieldans@gmail.com', 1999-10-23), (9, 'niella', 'zarck', 70000, 'niella@hotmail.com', 2000-09-04),(10 'abdias arsene', 'zountcheme' 100000, 'abdiasarsene@gmail.com', 2001-10-27)