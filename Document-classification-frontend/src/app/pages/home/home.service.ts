import { Injectable } from '@angular/core';
 
import { HttpClient, HttpParams, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { map, catchError} from 'rxjs/operators';
import { environment } from 'src/environments/environment';
 
@Injectable()
export class HomeService {
 
  API_URL: string = environment.API_URL;
 
  constructor(private http: HttpClient) {
  }
 
  getTags(): Observable<any> {
    return this.http.get(this.API_URL + 'tags')
  }

  getWordCloud(clusterID:number): Observable<any> {
    let params = new HttpParams()
    params = params.append('clusterID',clusterID)

    return this.http.get(this.API_URL + 'wordCloudByCluster',{params: params})
  }

  getInitialClusterList(): Observable<any> {
    // let params = new HttpParams()
    // params = params.append('clusterID',clusterID)
    return this.http.get(this.API_URL + 'initialCluster')
  }

  tagCluster(clusterID:number,tag:string): Observable<any> {
    let params = new HttpParams()
    params = params.append('clusterID',clusterID)
    params = params.append('tag',tag)
    return this.http.post(this.API_URL + 'taggedClusters',params,{responseType: 'json'})
  }

  addTag(tag:string): Observable<any> {
    let params = new HttpParams()
    params = params.append('tag',tag)
    return this.http.post(this.API_URL + 'add_tag',params,{responseType: 'json'})
  }
 
}