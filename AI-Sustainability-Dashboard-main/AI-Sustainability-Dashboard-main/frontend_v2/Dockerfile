FROM node:20-bullseye as build

WORKDIR /app

COPY package*.json /app/

RUN npm config delete proxy
RUN npm config delete http-proxy
RUN npm config delete https-proxy

RUN npm ci --prefer-offline

COPY . /app

RUN npm run build:production


#########################
#       PRODUCTION      #
#########################
FROM nginx:1-alpine as production

COPY --from=build /app/dist/browser /usr/share/nginx/html
COPY ./docker/nginx.conf /etc/nginx/nginx.conf

